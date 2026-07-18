# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import heapq
import json
import logging
import os
import platform
import signal
import threading
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

# SamplingParams.extra_args key used by the OPD teacher scorer.  Keeping the
# requested token ids on the request lets the vLLM GPU worker gather them before
# prompt logprobs are copied to CPU.
OPD_PROMPT_TOPK_EXTRA_ARGS_KEY = "verl_opd_prompt_topk"


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def monkey_patch_opd_prompt_topk_logprobs() -> None:
    """Gather OPD teacher logprobs on GPU instead of exporting the vocabulary.

    vLLM normally implements ``prompt_logprobs=-1`` by copying a
    ``[prompt_tokens, vocab_size]`` result to CPU and constructing Python
    ``Logprob`` objects.  OPD only needs the teacher's own top-k, the
    rollout-time student top-k ids, and the actually generated student token.
    This patch recognizes requests carrying :data:`OPD_PROMPT_TOPK_EXTRA_ARGS_KEY`
    and performs full-vocabulary logsumexp/top-k/gather on the GPU.  Only
    O(num_selected_tokens * k) values cross the GPU/CPU and worker boundaries.

    The non-OPD vLLM prompt-logprob path is left unchanged.
    """
    try:
        from vllm.v1.outputs import LogprobsTensors
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        # This optimization targets the vLLM V1 worker used by the native
        # teacher server.  Do not alter V0 behavior for unrelated rollouts.
        logger.warning("vLLM V1 GPUModelRunner is unavailable; OPD GPU top-k scoring patch was not installed")
        return

    method_name = "_get_prompt_logprobs_dict"
    original = getattr(GPUModelRunner, method_name, None)
    if original is None:
        logger.warning("vLLM GPUModelRunner has no %s; OPD GPU top-k scoring patch was not installed", method_name)
        return
    if getattr(original, "_verl_opd_gpu_topk", False):
        return

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        opd_request_ids = []
        regular_request_ids = []
        for req_id in num_prompt_logprobs_dict:
            request = self.requests[req_id]
            sampling_params = request.sampling_params
            extra_args = None if sampling_params is None else sampling_params.extra_args
            if extra_args and OPD_PROMPT_TOPK_EXTRA_ARGS_KEY in extra_args:
                opd_request_ids.append(req_id)
            else:
                regular_request_ids.append(req_id)

        if not opd_request_ids:
            return original(self, hidden_states, num_scheduled_tokens)
        if regular_request_ids:
            raise RuntimeError(
                "OPD compact prompt-logprob requests cannot share one vLLM prefill batch with regular "
                f"prompt-logprob requests; regular request ids: {regular_request_ids}."
            )

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict = {}
        completed_prefill_reqs = []
        logit_chunk_size = max(1, int(os.getenv("VERL_OPD_TOPK_LOGIT_CHUNK_SIZE", "64")))

        for req_id in opd_request_ids:
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                continue

            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                raise ValueError("OPD compact prompt logprobs require prompt token ids, not prompt embeddings.")
            metadata = request.sampling_params.extra_args[OPD_PROMPT_TOPK_EXTRA_ARGS_KEY]
            positions = [int(position) for position in metadata["positions"]]
            student_token_ids = [[int(token_id) for token_id in row] for row in metadata["token_ids"]]
            if len(positions) != len(student_token_ids):
                raise ValueError(
                    "OPD compact prompt-logprob positions and token-id rows must have equal length, got "
                    f"{len(positions)} and {len(student_token_ids)}."
                )
            if not positions:
                raise ValueError("OPD compact prompt-logprob request has no selected positions.")
            if positions != sorted(set(positions)):
                raise ValueError("OPD compact prompt-logprob positions must be sorted and unique.")
            topk = len(student_token_ids[0])
            if topk <= 0 or any(len(row) != topk for row in student_token_ids):
                raise ValueError("OPD compact prompt-logprob token-id rows must have one common positive top-k size.")
            if request.sampling_params.prompt_logprobs != 2 * topk:
                raise ValueError(
                    "OPD compact prompt-logprob requests must reserve teacher top-k plus student top-k slots: "
                    f"prompt_logprobs={request.sampling_params.prompt_logprobs}, topk={topk}."
                )

            num_prompt_tokens = len(request.prompt_token_ids)
            if positions[0] < 0 or positions[-1] >= num_prompt_tokens - 1:
                raise ValueError(
                    "OPD compact prompt-logprob position is outside the predictor range "
                    f"[0, {num_prompt_tokens - 1}): {positions}."
                )

            compact_logprobs = in_progress_dict.get(req_id)
            if compact_logprobs is None:
                # Column 0 is the actual next prompt token (vLLM convention),
                # followed by teacher top-k and rollout-time student top-k.
                compact_logprobs = LogprobsTensors.empty_cpu(len(positions), 2 * topk + 1)
                in_progress_dict[req_id] = compact_logprobs

            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = compact_logprobs

            if num_logits <= 0:
                continue

            selected_compact_rows = [
                compact_row
                for compact_row, position in enumerate(positions)
                if start_idx <= position < start_idx + num_logits
            ]
            if not selected_compact_rows:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            hidden_offset = self.query_start_loc.np[req_idx].item()
            for chunk_start in range(0, len(selected_compact_rows), logit_chunk_size):
                compact_rows = selected_compact_rows[chunk_start : chunk_start + logit_chunk_size]
                absolute_positions = [positions[row] for row in compact_rows]
                hidden_indices = torch.tensor(
                    [hidden_offset + position - start_idx for position in absolute_positions],
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                selected_hidden_states = hidden_states.index_select(0, hidden_indices)
                # float32 logsumexp matches full-vocabulary probabilities while
                # avoiding bf16 underflow in small-probability tails.
                logits = self.model.compute_logits(selected_hidden_states).float()
                log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
                teacher_logits, teacher_ids = torch.topk(logits, k=topk, dim=-1)

                student_ids = torch.tensor(
                    [student_token_ids[row] for row in compact_rows],
                    dtype=torch.int64,
                    device=logits.device,
                )
                if torch.any(student_ids < 0) or torch.any(student_ids >= logits.shape[-1]):
                    raise ValueError("OPD student top-k contains a token id outside the teacher vocabulary.")
                student_logits = torch.gather(logits, dim=-1, index=student_ids)
                actual_ids = torch.tensor(
                    [request.prompt_token_ids[position + 1] for position in absolute_positions],
                    dtype=torch.int64,
                    device=logits.device,
                )
                actual_logits = torch.gather(logits, dim=-1, index=actual_ids.unsqueeze(-1)).squeeze(-1)
                actual_ranks = (logits > actual_logits.unsqueeze(-1)).sum(dim=-1, dtype=torch.int32) + 1

                output_ids = torch.cat(
                    [actual_ids.unsqueeze(-1), teacher_ids.to(torch.int64), student_ids], dim=-1
                )
                output_logprobs = torch.cat(
                    [
                        actual_logits.unsqueeze(-1) - log_normalizer,
                        teacher_logits - log_normalizer,
                        student_logits - log_normalizer,
                    ],
                    dim=-1,
                )
                cpu_rows = torch.tensor(compact_rows, dtype=torch.int64)
                compact_logprobs.logprob_token_ids.index_copy_(
                    0, cpu_rows, output_ids.to(device="cpu", dtype=torch.int32)
                )
                compact_logprobs.logprobs.index_copy_(
                    0, cpu_rows, output_logprobs.to(device="cpu", dtype=torch.float32)
                )
                compact_logprobs.selected_token_ranks.index_copy_(
                    0, cpu_rows, actual_ranks.to(device="cpu", dtype=torch.int32)
                )

        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        if prompt_logprobs_dict:
            self._sync_device()
        return prompt_logprobs_dict

    _get_prompt_logprobs_dict._verl_opd_gpu_topk = True
    setattr(GPUModelRunner, method_name, _get_prompt_logprobs_dict)
    logger.info("Installed compact GPU OPD teacher prompt-logprob scoring for vLLM V1")


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # Keep full-vocabulary teacher logits on GPU for OPD scoring and only
        # export teacher/student top-k values.
        monkey_patch_opd_prompt_topk_logprobs()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def _get_drafter_model(self):
        """Return the drafter's model object, or None if unavailable."""
        drafter = getattr(self.model_runner, "drafter", None)
        return drafter.model if drafter is not None and hasattr(drafter, "model") else None

    def enable_opd_prompt_topk_logprobs(self):
        """Install the compact OPD prompt-logprob path on this vLLM worker."""
        monkey_patch_opd_prompt_topk_logprobs()

    def _get_draft_model_config(self):
        """Return the draft model config from speculative_config, or None."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec.draft_model_config if spec is not None and spec.draft_model_config is not None else None

    def _use_mtp_drafter_weight_sync(self):
        """Return whether the vLLM MTP drafter should receive actor weights."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec is not None and spec.method == "mtp" and self._get_drafter_model() is not None

    def _iter_all_models(self):
        """Yield models that need weight updates.

        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
        draft models are not compatible with actor weight loading through this path.
        """
        yield self.model_runner.model
        if self._use_mtp_drafter_weight_sync():
            yield self._get_drafter_model()

    def _iter_all_models_with_config(self):
        """Yield (model, model_config) for models that need post-processing."""
        yield self.model_runner.model, self.model_runner.vllm_config.model_config
        if self._use_mtp_drafter_weight_sync():
            draft_cfg = self._get_draft_model_config()
            if draft_cfg is not None:
                yield self._get_drafter_model(), draft_cfg

    def monkey_patch_model(self, vocab_size: int):
        for model in self._iter_all_models():
            # patch compute_logits to avoid sampling OOV token
            monkey_patch_compute_logits(model, vocab_size)
            # patch weight loader to support MoE model
            patch_vllm_moe_model_weight_loader(model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            for model in self._iter_all_models():
                prepare_qat_for_load_weights(model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            for model in self._iter_all_models():
                patch_vllm_moe_model_weight_loader(model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            for model in self._iter_all_models():
                manual_process_weights_after_loading(model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
                # Keep the draft model in sync when present.
                if self._use_mtp_drafter_weight_sync():
                    load_quanted_weights(weights, self.model_runner, is_drafter=True)
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                for model in self._iter_all_models():
                    model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def _dense_full_vocab_logprobs(logprobs_dict: dict) -> list[float]:
    """Convert one vLLM all-logprobs dictionary to dense token-id order."""
    vocab_size = len(logprobs_dict)
    dense_logprobs = [None] * vocab_size
    for token_id, token_logprob in logprobs_dict.items():
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                "Full-vocabulary logprobs require contiguous token ids in [0, vocab_size), "
                f"got token id {token_id} for vocab size {vocab_size}."
            )
        dense_logprobs[token_id] = token_logprob.logprob
    if any(value is None for value in dense_logprobs):
        raise ValueError("vLLM full-vocabulary logprobs omitted one or more token ids.")
    return dense_logprobs


def _lookup_logprob_entry(logprobs_dict: dict, token_id: int):
    entry = logprobs_dict.get(token_id)
    if entry is None:
        entry = logprobs_dict.get(str(token_id))
    return entry


def _topk_ids_from_logprobs_dict(logprobs_dict: dict, k: int) -> list[int]:
    """Return the token ids with the k largest log-probabilities from one vLLM logprob row."""
    if k <= 0:
        return []
    top_entries = heapq.nlargest(
        k,
        ((float(token_logprob.logprob), int(token_id)) for token_id, token_logprob in logprobs_dict.items()),
        key=lambda item: item[0],
    )
    if len(top_entries) != k:
        raise ValueError(f"vLLM returned only {len(top_entries)} prompt logprobs, cannot build teacher top-{k}.")
    return [token_id for _logprob, token_id in top_entries]


def extract_sample_logprobs(output: RequestOutput, num_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract generated-token log probabilities when requested."""
    if num_logprobs is None:
        return

    if num_logprobs == -1:
        result_dict["sample_full_logprobs"] = [
            _dense_full_vocab_logprobs(logprobs_dict) for logprobs_dict in output.outputs[0].logprobs
        ]
        return

    if num_logprobs <= 0:
        return

    sample_ids: list[list[int]] = []
    sample_logprobs: list[list[float]] = []
    for position, logprobs_dict in enumerate(output.outputs[0].logprobs):
        ids = [None] * num_logprobs
        logprobs = [None] * num_logprobs
        for token_id, token_logprob in logprobs_dict.items():
            rank = token_logprob.rank
            if rank > num_logprobs:
                continue
            ids[rank - 1] = int(token_id)
            logprobs[rank - 1] = float(token_logprob.logprob)
        if any(token_id is None for token_id in ids) or any(logprob is None for logprob in logprobs):
            raise ValueError(
                f"vLLM returned incomplete generated-token top-{num_logprobs} logprobs at position {position}."
            )
        sample_ids.append(ids)
        sample_logprobs.append(logprobs)
    result_dict["sample_ids"] = sample_ids
    result_dict["sample_logprobs"] = sample_logprobs


def extract_prompt_selected_logprobs(
    output: RequestOutput,
    prompt_logprob_token_ids: Optional[dict],
    result_dict: dict[str, list],
):
    """Extract prompt logprobs only for caller-specified token ids at selected positions."""
    if prompt_logprob_token_ids is None:
        return

    positions = [int(position) for position in prompt_logprob_token_ids.get("positions", [])]
    token_ids = prompt_logprob_token_ids.get("token_ids", [])
    if len(positions) != len(token_ids):
        raise ValueError(
            "prompt_logprob_token_ids must contain equal-length 'positions' and 'token_ids' lists, "
            f"got {len(positions)} and {len(token_ids)}."
        )

    if prompt_logprob_token_ids.get("gpu_compact", False):
        # The GPU worker returned only the requested positions.  Each row
        # contains the actual student token, teacher top-k, and student top-k;
        # full-vocabulary logprobs never reached this process.
        compact_logprobs = output.prompt_logprobs
        actual_token_ids = [int(token_id) for token_id in prompt_logprob_token_ids.get("actual_token_ids", [])]
        if not all(
            hasattr(compact_logprobs, field)
            for field in ("start_indices", "end_indices", "token_ids", "logprobs")
        ):
            raise TypeError("Compact GPU prompt logprobs require vLLM flat_logprobs output.")
        # FlatLogprobs row zero is the conventional None entry for the first
        # prompt token.  The remaining rows correspond exactly to `positions`.
        if len(compact_logprobs) != len(positions) + 1 or len(actual_token_ids) != len(positions):
            raise ValueError(
                "Compact GPU prompt logprobs must return exactly one row per requested position, got "
                f"rows={len(compact_logprobs) - 1}, positions={len(positions)}, "
                f"actual_ids={len(actual_token_ids)}."
            )

        selected_ids: list[list[int]] = []
        selected_logprobs: list[list[float]] = []
        teacher_topk_ids: list[list[int]] = []
        teacher_topk_logprobs: list[list[float]] = []
        actual_logprobs: list[float] = []
        for compact_row, (position, row_token_ids, actual_token_id) in enumerate(
            zip(positions, token_ids, actual_token_ids, strict=True), start=1
        ):
            requested_ids = [int(token_id) for token_id in row_token_ids]
            topk = len(requested_ids)
            start = compact_logprobs.start_indices[compact_row]
            end = compact_logprobs.end_indices[compact_row]
            flat_ids = [int(token_id) for token_id in compact_logprobs.token_ids[start:end]]
            flat_values = [float(logprob) for logprob in compact_logprobs.logprobs[start:end]]
            if len(flat_ids) != 2 * topk + 1:
                raise ValueError(
                    "Compact GPU prompt-logprob row must contain actual, teacher top-k, and student top-k; "
                    f"got width={len(flat_ids)}, expected={2 * topk + 1}, position={position}."
                )
            returned_actual_id = flat_ids[0]
            row_teacher_topk_ids = flat_ids[1 : topk + 1]
            row_teacher_topk_logprobs = flat_values[1 : topk + 1]
            returned_selected_ids = flat_ids[topk + 1 :]
            row_selected_logprobs = flat_values[topk + 1 :]
            if returned_actual_id != actual_token_id:
                raise ValueError(
                    "Compact GPU actual token is not aligned with the prompt: "
                    f"returned={returned_actual_id}, expected={actual_token_id}, position={position}."
                )
            if returned_selected_ids != requested_ids:
                raise ValueError(
                    "Compact GPU student top-k ids differ from the requested rollout-time ids: "
                    f"returned={returned_selected_ids}, expected={requested_ids}, position={position}."
                )

            selected_ids.append(returned_selected_ids)
            selected_logprobs.append(row_selected_logprobs)
            teacher_topk_ids.append(row_teacher_topk_ids)
            teacher_topk_logprobs.append(row_teacher_topk_logprobs)
            actual_logprobs.append(flat_values[0])

        result_dict["prompt_selected_positions"] = positions
        result_dict["prompt_selected_ids"] = selected_ids
        result_dict["prompt_selected_logprobs"] = selected_logprobs
        result_dict["prompt_teacher_topk_ids"] = teacher_topk_ids
        result_dict["prompt_teacher_topk_logprobs"] = teacher_topk_logprobs
        result_dict["prompt_selected_actual_ids"] = actual_token_ids
        result_dict["prompt_selected_actual_logprobs"] = actual_logprobs
        return

    prompt_rows = output.prompt_logprobs[1:]
    selected_ids: list[list[int]] = []
    selected_logprobs: list[list[float]] = []
    teacher_topk_ids: list[list[int]] = []
    teacher_topk_logprobs: list[list[float]] = []
    for position, row_token_ids in zip(positions, token_ids, strict=True):
        if position < 0 or position >= len(prompt_rows):
            raise ValueError(
                f"Requested prompt logprob position {position} outside available range [0, {len(prompt_rows)})."
            )
        logprobs_dict = prompt_rows[position]
        row_ids = [int(token_id) for token_id in row_token_ids]
        row_teacher_topk_ids = _topk_ids_from_logprobs_dict(logprobs_dict, len(row_ids))
        row_teacher_topk_logprobs: list[float] = []
        for token_id in row_teacher_topk_ids:
            token_logprob = _lookup_logprob_entry(logprobs_dict, token_id)
            if token_logprob is None:
                raise ValueError(
                    "vLLM did not return a teacher top-k prompt token id. "
                    f"missing token_id={token_id} at position={position}."
                )
            row_teacher_topk_logprobs.append(float(token_logprob.logprob))
        teacher_topk_ids.append(row_teacher_topk_ids)
        teacher_topk_logprobs.append(row_teacher_topk_logprobs)
        row_logprobs: list[float] = []
        for token_id in row_ids:
            token_logprob = _lookup_logprob_entry(logprobs_dict, token_id)
            if token_logprob is None:
                raise ValueError(
                    "vLLM did not return a requested prompt token id. "
                    "Use the compact GPU prompt-logprob path for student-top-k OPD; "
                    f"missing token_id={token_id} at position={position}."
                )
            row_logprobs.append(float(token_logprob.logprob))
        selected_ids.append(row_ids)
        selected_logprobs.append(row_logprobs)

    result_dict["prompt_selected_positions"] = positions
    result_dict["prompt_selected_ids"] = selected_ids
    result_dict["prompt_selected_logprobs"] = selected_logprobs
    result_dict["prompt_teacher_topk_ids"] = teacher_topk_ids
    result_dict["prompt_teacher_topk_logprobs"] = teacher_topk_logprobs


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    if num_prompt_logprobs == -1:
        prompt_full_logprobs = [
            _dense_full_vocab_logprobs(logprobs_dict) for logprobs_dict in output.prompt_logprobs[1:]
        ]

        if not prompt_full_logprobs:
            raise ValueError("vLLM returned no prompt positions for full-vocabulary log-prob extraction.")
        # The final row predicts the generated dummy token and is not used for
        # response-token distillation, but retaining it preserves sequence length.
        vocab_size = len(prompt_full_logprobs[0])
        if any(len(row) != vocab_size for row in prompt_full_logprobs):
            raise ValueError("vLLM returned inconsistent vocabulary widths across prompt positions.")
        prompt_full_logprobs.append([0.0] * vocab_size)
        result_dict["prompt_full_logprobs"] = prompt_full_logprobs
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls

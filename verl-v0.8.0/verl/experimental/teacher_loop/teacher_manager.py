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
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    if distillation_loss_config.loss_settings.use_full_vocab:
        # The OPD "full-vocab" path now uses only the original student-selected
        # token at each trainable response position.  Requesting
        # prompt_logprobs=0 asks the server for exactly that token logprob.
        num_logprobs = 0
    elif distillation_loss_config.loss_settings.use_topk:
        num_logprobs = distillation_loss_config.topk
    else:
        num_logprobs = 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def align_teacher_outputs_to_student_response(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    teacher_prompt_length: int,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher distributions to the student's response positions.

    Teacher prompt-logprob row ``i`` is the distribution after token ``i`` and
    therefore predicts token ``i + 1``.  A response of length ``R`` uses rows
    ``prompt_length - 1 : prompt_length + R - 1``.  When teacher and student
    prompt lengths differ, this function moves exactly those rows to the
    equivalent positions in the student sequence.  Prompt rows are irrelevant to
    the distillation loss and are filled with padding values.
    """
    if teacher_prompt_length <= 0 or student_prompt_length <= 0:
        raise ValueError(
            "Teacher and student prompts must contain at least one token to align next-token distributions."
        )
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}.")

    teacher_sequence_length = teacher_prompt_length + response_length
    if teacher_ids.shape[0] != teacher_sequence_length or teacher_logprobs.shape[0] != teacher_sequence_length:
        raise ValueError(
            "Teacher output length must match teacher_prompt_length + response_length, "
            f"got ids={teacher_ids.shape[0]}, logprobs={teacher_logprobs.shape[0]}, "
            f"expected={teacher_sequence_length}."
        )
    if teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError(
            f"Teacher ids and logprobs must have identical shapes, got {teacher_ids.shape} and "
            f"{teacher_logprobs.shape}."
        )

    # Preserve the native path byte-for-byte when both prompt prefixes match in length.
    if teacher_prompt_length == student_prompt_length:
        return teacher_ids, teacher_logprobs

    student_sequence_length = student_prompt_length + response_length
    aligned_ids = torch.full(
        (student_sequence_length, *teacher_ids.shape[1:]),
        fill_value=pad_token_id,
        dtype=teacher_ids.dtype,
        device=teacher_ids.device,
    )
    aligned_logprobs = torch.zeros(
        (student_sequence_length, *teacher_logprobs.shape[1:]),
        dtype=teacher_logprobs.dtype,
        device=teacher_logprobs.device,
    )

    teacher_start = teacher_prompt_length - 1
    student_start = student_prompt_length - 1
    aligned_ids[student_start : student_start + response_length] = teacher_ids[
        teacher_start : teacher_start + response_length
    ]
    aligned_logprobs[student_start : student_start + response_length] = teacher_logprobs[
        teacher_start : teacher_start + response_length
    ]
    return aligned_ids, aligned_logprobs


def select_teacher_full_vocab_response_rows(
    teacher_full_logprobs: torch.Tensor,
    *,
    teacher_prompt_length: int,
    response_mask: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select full-vocabulary rows that predict trainable response tokens.

    Returned indices are response-local, so teacher and student prompt lengths may
    differ without any absolute full-sequence alignment. Environment-observation
    rows and all masked rows are discarded before transfer to the actor.
    """
    if teacher_prompt_length <= 0:
        raise ValueError("Teacher prompts must contain at least one token for next-token scoring.")
    response_length = len(response_mask)
    expected_length = teacher_prompt_length + response_length
    if teacher_full_logprobs.dim() != 2 or teacher_full_logprobs.shape[0] != expected_length:
        raise ValueError(
            "Full-vocabulary teacher output must have shape [teacher_prompt_length + response_length, vocab], "
            f"got {teacher_full_logprobs.shape}, expected first dimension {expected_length}."
        )

    response_indices = torch.tensor(
        [index for index, mask_value in enumerate(response_mask) if bool(mask_value)], dtype=torch.int64
    )
    if response_indices.numel() == 0:
        return response_indices, teacher_full_logprobs.new_empty((0, teacher_full_logprobs.shape[-1]))
    teacher_predictor_rows = teacher_prompt_length - 1 + response_indices
    return response_indices, teacher_full_logprobs.index_select(0, teacher_predictor_rows)


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    @property
    def uses_full_vocab(self) -> bool:
        return self.distillation_loss_config.loss_settings.use_full_vocab

    @property
    def uses_student_topk_support(self) -> bool:
        return (
            self.distillation_loss_config.loss_mode == "reverse_kl_topk"
            and self.distillation_loss_config.loss_settings.use_topk
        )

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    def empty_teacher_outputs(self, *, sequence_length: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build shape-compatible teacher tensors for a trajectory excluded from OPD."""
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
        width = (
            int(self.distillation_loss_config.topk or 1)
            if self.distillation_loss_config.loss_settings.use_topk
            else 1
        )
        teacher_ids = torch.full((sequence_length, width), pad_token_id, dtype=torch.int32)
        teacher_logprobs = torch.zeros((sequence_length, width), dtype=torch.float32)
        return teacher_ids, teacher_logprobs

    def empty_teacher_full_vocab_outputs(self, *, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build zero-row selected-token teacher tensors for a trajectory excluded from OPD.

        ``vocab_size`` is kept in the signature for compatibility with the old
        full-vocabulary path; selected-token OPD stores one logprob per row.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        return torch.empty((0,), dtype=torch.int64), torch.empty((0, 1), dtype=torch.float32)

    def empty_teacher_student_topk_outputs(
        self, *, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build zero-row sparse teacher tensors for student-top-k reverse KL."""
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")
        return (
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, topk), dtype=torch.int64),
            torch.empty((0, topk), dtype=torch.float32),
            torch.empty((0, topk), dtype=torch.int64),
        )

    async def generate_teacher_critique_single(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        routing_key: Optional[str] = None,
    ) -> list[int]:
        """Generate privileged critique ``c`` with the configured native teacher server."""
        if not prompt_ids:
            raise ValueError("Teacher critique prompt must not be empty.")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
        teacher_key = self._resolve_teacher_key(routing_key)
        client = self.teacher_client[teacher_key]
        output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params={
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        )
        return list(output.token_ids)

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_full_vocab_logprobs_single(
        self,
        *,
        sequence_ids: list[int],
        teacher_prompt_length: int,
        response_mask: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher logprobs for the original student-selected response tokens only."""
        if not self.uses_full_vocab:
            raise RuntimeError("compute_teacher_full_vocab_logprobs_single requires the OPD selected-token mode.")
        if len(sequence_ids) != teacher_prompt_length + len(response_mask):
            raise ValueError(
                "Teacher sequence must be teacher prompt plus the exact student response, got "
                f"sequence={len(sequence_ids)}, prompt={teacher_prompt_length}, response={len(response_mask)}."
            )

        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        if teacher_model_config.inference.temperature != 1.0:
            raise NotImplementedError("Teacher selected-token prompt_logprobs require teacher temperature 1.0.")
        client = self.teacher_client[teacher_key]
        response_indices = torch.tensor(
            [index for index, mask_value in enumerate(response_mask) if bool(mask_value)], dtype=torch.int64
        )
        if response_indices.numel() == 0:
            raise ValueError(
                "Teacher selected-token scoring received no trainable response tokens. "
                "Excluded OPD trajectories should use empty_teacher_full_vocab_outputs instead."
            )

        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": teacher_model_config.inference.temperature,
                "prompt_logprobs": 0,
            },
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int64)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"], dtype=torch.float32)
        if teacher_ids.shape != teacher_logprobs.shape or teacher_ids.shape != (len(sequence_ids), 1):
            raise ValueError(
                "Teacher selected-token prompt_logprobs must have shape [teacher_prompt + response, 1], "
                f"got ids={teacher_ids.shape}, logprobs={teacher_logprobs.shape}."
            )

        teacher_predictor_rows = teacher_prompt_length - 1 + response_indices
        selected_teacher_ids = teacher_ids.index_select(0, teacher_predictor_rows).squeeze(-1)
        selected_response_ids = torch.tensor(
            [sequence_ids[teacher_prompt_length + index] for index in response_indices.tolist()],
            dtype=torch.int64,
        )
        if not torch.equal(selected_teacher_ids.cpu(), selected_response_ids):
            raise ValueError(
                "Teacher selected-token prompt_logprobs are not aligned with the original student response tokens: "
                f"teacher_ids={selected_teacher_ids.tolist()}, response_ids={selected_response_ids.tolist()}."
            )

        selected_teacher_logprobs = teacher_logprobs.index_select(0, teacher_predictor_rows)
        return response_indices, selected_teacher_logprobs

    async def compute_teacher_student_topk_logprobs_single(
        self,
        *,
        sequence_ids: list[int],
        teacher_prompt_length: int,
        response_mask: list[int],
        student_topk_ids: torch.Tensor,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute teacher logprobs only for rollout-time student top-k token ids."""
        if not self.uses_student_topk_support:
            raise RuntimeError(
                "compute_teacher_student_topk_logprobs_single requires reverse_kl_topk student-top-k mode."
            )
        if len(sequence_ids) != teacher_prompt_length + len(response_mask):
            raise ValueError(
                "Teacher sequence must be teacher prompt plus the exact student response, got "
                f"sequence={len(sequence_ids)}, prompt={teacher_prompt_length}, response={len(response_mask)}."
            )

        response_indices = torch.tensor(
            [index for index, mask_value in enumerate(response_mask) if bool(mask_value)], dtype=torch.int64
        )
        if response_indices.numel() == 0:
            raise ValueError(
                "Teacher student-top-k scoring received no trainable response tokens. "
                "Excluded OPD trajectories should use empty_teacher_student_topk_outputs instead."
            )
        student_topk_ids = student_topk_ids.to(dtype=torch.int64, device="cpu")
        if student_topk_ids.dim() != 2 or student_topk_ids.shape[0] != response_indices.numel():
            raise ValueError(
                "student_topk_ids must have shape [num_trainable_response_tokens, topk], got "
                f"{student_topk_ids.shape} for {response_indices.numel()} response tokens."
            )

        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        if teacher_model_config.inference.temperature != 1.0:
            raise NotImplementedError("Teacher selected prompt logprobs require teacher temperature 1.0.")
        client = self.teacher_client[teacher_key]
        teacher_predictor_rows = teacher_prompt_length - 1 + response_indices
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": teacher_model_config.inference.temperature,
                "prompt_logprobs": -1,
                "prompt_logprob_token_ids": {
                    "positions": teacher_predictor_rows.tolist(),
                    "token_ids": student_topk_ids.tolist(),
                },
            },
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        returned_ids = torch.tensor(teacher_output.extra_fields["prompt_selected_ids"], dtype=torch.int64)
        teacher_topk_logprobs = torch.tensor(
            teacher_output.extra_fields["prompt_selected_logprobs"], dtype=torch.float32
        )
        teacher_topk_ids = torch.tensor(
            teacher_output.extra_fields["prompt_teacher_topk_ids"], dtype=torch.int64
        )
        if returned_ids.shape != student_topk_ids.shape or not torch.equal(returned_ids, student_topk_ids):
            raise ValueError(
                "Teacher selected prompt logprobs were not returned for the requested student top-k ids: "
                f"returned={returned_ids.shape}, requested={student_topk_ids.shape}."
            )
        if teacher_topk_logprobs.shape != student_topk_ids.shape:
            raise ValueError(
                "Teacher selected prompt logprobs must match student_topk_ids shape, got "
                f"{teacher_topk_logprobs.shape} vs {student_topk_ids.shape}."
            )
        if teacher_topk_ids.shape != student_topk_ids.shape:
            raise ValueError(
                "Teacher top-k ids must match student_topk_ids shape, got "
                f"{teacher_topk_ids.shape} vs {student_topk_ids.shape}."
            )
        return response_indices, student_topk_ids, teacher_topk_logprobs, teacher_topk_ids

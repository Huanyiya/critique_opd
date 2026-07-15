# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _prompt_lengths(data: TensorDict) -> torch.Tensor:
    prompts = data["prompts"]
    if prompts.is_nested:
        return prompts.offsets().diff()

    attention_mask = data.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.is_nested:
            raise ValueError("A regular prompt tensor cannot be paired with a nested attention_mask.")
        return attention_mask[:, : prompts.shape[1]].sum(dim=-1)

    # TransferQueue stores unpadded per-sample tensors. If every prompt in a
    # fetched microbatch happens to have the same length it may stack them into a
    # regular tensor, in which case the width is the true prompt length.
    return torch.full((prompts.shape[0],), prompts.shape[1], dtype=torch.int64, device=prompts.device)


def compute_reverse_kl_full_vocab(
    student_logits: torch.Tensor,
    teacher_full_log_probs: torch.Tensor,
    teacher_response_indices: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
) -> dict[str, torch.Tensor]:
    """Compute selected-token OPD on the original student response tokens.

    This keeps the historical ``reverse_kl_full_vocab`` entry point, but no
    longer reduces over the whole vocabulary.  Teacher rows are stored only for
    response positions whose ``response_mask`` is one, and each row contains the
    teacher logprob of the exact token sampled by the student at that position.
    """
    if get_ulysses_sequence_parallel_world_size() != 1:
        raise NotImplementedError(
            "reverse_kl_full_vocab selected-token OPD does not yet support Ulysses sequence parallelism; "
            "set actor_rollout_ref.actor.ulysses_sequence_parallel_size=1."
        )
    if not teacher_full_log_probs.is_nested or not teacher_response_indices.is_nested:
        raise ValueError("Selected-token teacher log probabilities and response indices must be nested tensors.")
    if student_logits.shape[0] != 1:
        raise ValueError(f"Expected flattened FSDP logits with batch dimension 1, got {student_logits.shape}.")

    student_logits_flat = student_logits.squeeze(0)
    input_ids_flat = data["input_ids"].values()
    input_offsets = data["input_ids"].offsets()
    prompt_lengths = _prompt_lengths(data).to(device=input_offsets.device, dtype=torch.int64)
    teacher_rows = teacher_full_log_probs.unbind()
    response_indices = teacher_response_indices.unbind()
    if not (len(teacher_rows) == len(response_indices) == prompt_lengths.numel()):
        raise ValueError(
            "Teacher selected-token batch size must match the student batch: "
            f"rows={len(teacher_rows)}, indices={len(response_indices)}, batch={prompt_lengths.numel()}."
        )

    predictor_positions = []
    selected_teacher_logprobs = []
    for sample_idx, (sample_teacher, sample_indices) in enumerate(zip(teacher_rows, response_indices, strict=True)):
        if sample_teacher.shape[0] != sample_indices.numel():
            raise ValueError(
                f"Sample {sample_idx} has {sample_teacher.shape[0]} teacher rows but "
                f"{sample_indices.numel()} response indices."
            )
        if sample_teacher.dim() != 2 or sample_teacher.shape[-1] != 1:
            raise ValueError(
                "Selected-token teacher log probabilities must have shape [num_selected_tokens, 1], "
                f"got {sample_teacher.shape} for sample {sample_idx}."
            )
        if sample_indices.numel() == 0:
            continue
        response_length = input_offsets[sample_idx + 1] - input_offsets[sample_idx] - prompt_lengths[sample_idx]
        sample_indices_device = sample_indices.to(device=input_offsets.device, dtype=torch.int64)
        if bool(((sample_indices_device < 0) | (sample_indices_device >= response_length)).any()):
            raise ValueError(
                f"Sample {sample_idx} contains response indices outside [0, {response_length.item()}): "
                f"{sample_indices.tolist()}."
            )
        predictor_positions.append(
            input_offsets[sample_idx] + prompt_lengths[sample_idx] - 1 + sample_indices_device
        )
        selected_teacher_logprobs.append(sample_teacher.squeeze(-1))

    # Keep an empty OPD microbatch connected to the current forward pass.
    if not predictor_positions:
        return {"distillation_losses": (student_logits_flat.sum(dim=-1) * 0.0).unsqueeze(0)}

    predictor_positions = torch.cat(predictor_positions)
    teacher_token_log_probs = torch.cat(selected_teacher_logprobs, dim=0).to(student_logits_flat.device).float()
    selected_token_ids = input_ids_flat.index_select(0, predictor_positions + 1).to(
        device=student_logits_flat.device,
        dtype=torch.int64,
    )
    student_token_log_probs = F.log_softmax(
        student_logits_flat.index_select(0, predictor_positions).float(),
        dim=-1,
    ).gather(dim=-1, index=selected_token_ids.unsqueeze(-1)).squeeze(-1)
    if teacher_token_log_probs.shape != student_token_log_probs.shape:
        raise ValueError(
            f"Teacher/student selected-token rows do not align: {teacher_token_log_probs.shape} vs "
            f"{student_token_log_probs.shape}."
        )

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        teacher_token_log_probs = teacher_token_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        student_token_log_probs = student_token_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    reverse_kl = student_token_log_probs.exp() * (student_token_log_probs - teacher_token_log_probs)

    token_losses = student_logits_flat.sum(dim=-1) * 0.0
    token_losses = token_losses.index_copy(0, predictor_positions, reverse_kl)
    return {"distillation_losses": token_losses.unsqueeze(0)}


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_full_log_probs: torch.Tensor,
    teacher_response_indices: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str,
    unprivileged_teacher_full_log_probs: torch.Tensor | None = None,
    unprivileged_teacher_response_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute reverse KL on the current student's top-k support.

    The privileged loss is returned as ``distillation_losses`` and drives the
    update. When the unprivileged current-prompt teacher tensors are present, a
    second reverse KL is returned under ``unprivileged_*`` keys for comparison.

    The top-k IDs are selected from the current student logits during this actor
    forward.  Teacher scoring happens earlier in the rollout pipeline, so the
    teacher provides dense response-position log probabilities and this loss
    gathers the teacher probabilities at the student-selected IDs.
    """
    del data_format
    if get_ulysses_sequence_parallel_world_size() != 1:
        raise NotImplementedError(
            "reverse_kl_topk with student-selected support does not yet support Ulysses sequence parallelism; "
            "set actor_rollout_ref.actor.ulysses_sequence_parallel_size=1."
        )
    if student_logits.shape[0] != 1:
        raise ValueError(f"Expected flattened FSDP logits with batch dimension 1, got {student_logits.shape}.")

    student_logits_flat = student_logits.squeeze(0)
    input_offsets = data["input_ids"].offsets()
    prompt_lengths = _prompt_lengths(data).to(device=input_offsets.device, dtype=torch.int64)
    loss_config: DistillationLossConfig = config.distillation_loss
    topk = int(loss_config.topk)
    if topk <= 0:
        raise ValueError(f"reverse_kl_topk requires a positive topk value, got {loss_config.topk}.")
    if topk > student_logits_flat.shape[-1]:
        raise ValueError(f"topk={topk} exceeds vocabulary size {student_logits_flat.shape[-1]}.")

    def _empty_outputs() -> dict[str, torch.Tensor]:
        zero = (student_logits_flat.sum(dim=-1) * 0.0).unsqueeze(0)
        return {
            "distillation_losses": zero,
            "student_mass": zero.detach().clone(),
            "teacher_mass": zero.detach().clone(),
            "overlap_count": zero.detach().clone(),
            "overlap_token_advantage": zero.detach().clone(),
        }

    def _collect_teacher_rows(
        full_log_probs: torch.Tensor,
        response_indices: torch.Tensor,
        *,
        expected_response_indices: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[torch.Tensor]]:
        if not full_log_probs.is_nested or not response_indices.is_nested:
            raise ValueError("Dense teacher log probabilities and response indices must be nested tensors.")
        teacher_rows = full_log_probs.unbind()
        index_rows = response_indices.unbind()
        if not (len(teacher_rows) == len(index_rows) == prompt_lengths.numel()):
            raise ValueError(
                "Teacher dense-logprob batch size must match the student batch: "
                f"rows={len(teacher_rows)}, indices={len(index_rows)}, batch={prompt_lengths.numel()}."
            )

        predictor_positions: list[torch.Tensor] = []
        selected_teacher_rows: list[torch.Tensor] = []
        collected_indices: list[torch.Tensor] = []
        for sample_idx, (sample_teacher, sample_indices) in enumerate(zip(teacher_rows, index_rows, strict=True)):
            if sample_teacher.shape[0] != sample_indices.numel():
                raise ValueError(
                    f"Sample {sample_idx} has {sample_teacher.shape[0]} teacher rows but "
                    f"{sample_indices.numel()} response indices."
                )
            if sample_teacher.dim() != 2 or sample_teacher.shape[-1] != student_logits_flat.shape[-1]:
                raise ValueError(
                    "Teacher dense log probabilities must have shape [num_selected_tokens, vocab], "
                    f"got {sample_teacher.shape} for sample {sample_idx}, "
                    f"student vocab={student_logits_flat.shape[-1]}."
                )
            sample_indices_device = sample_indices.to(device=input_offsets.device, dtype=torch.int64)
            if expected_response_indices is not None:
                expected = expected_response_indices[sample_idx].to(device=input_offsets.device, dtype=torch.int64)
                if sample_indices_device.shape != expected.shape or not torch.equal(sample_indices_device, expected):
                    raise ValueError(
                        "Privileged and unprivileged teacher rows must score the same response-local positions; "
                        f"sample {sample_idx} got {sample_indices_device.tolist()} vs expected {expected.tolist()}."
                    )
            collected_indices.append(sample_indices.detach().cpu())
            if sample_indices.numel() == 0:
                continue

            response_length = input_offsets[sample_idx + 1] - input_offsets[sample_idx] - prompt_lengths[sample_idx]
            if bool(((sample_indices_device < 0) | (sample_indices_device >= response_length)).any()):
                raise ValueError(
                    f"Sample {sample_idx} contains response indices outside [0, {response_length.item()}): "
                    f"{sample_indices.tolist()}."
                )

            predictor_positions.append(
                input_offsets[sample_idx] + prompt_lengths[sample_idx] - 1 + sample_indices_device
            )
            selected_teacher_rows.append(sample_teacher)

        if not predictor_positions:
            return None, None, collected_indices
        return (
            torch.cat(predictor_positions).to(device=student_logits_flat.device, dtype=torch.int64),
            torch.cat(selected_teacher_rows, dim=0).to(device=student_logits_flat.device).float(),
            collected_indices,
        )

    def _compute_on_student_support(
        full_log_probs: torch.Tensor,
        response_indices: torch.Tensor,
        *,
        expected_response_indices: list[torch.Tensor] | None = None,
        cached_support: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        predictor_positions, selected_teacher_full_log_probs, collected_indices = _collect_teacher_rows(
            full_log_probs,
            response_indices,
            expected_response_indices=expected_response_indices,
        )
        if predictor_positions is None or selected_teacher_full_log_probs is None:
            return _empty_outputs(), collected_indices, None

        if cached_support is None:
            selected_student_log_probs = F.log_softmax(
                student_logits_flat.index_select(0, predictor_positions).float(),
                dim=-1,
            )
            student_topk_log_probs, student_topk_ids = torch.topk(selected_student_log_probs, k=topk, dim=-1)
            support = (
                predictor_positions,
                student_topk_log_probs,
                student_topk_ids,
            )
        else:
            predictor_positions, student_topk_log_probs, student_topk_ids = cached_support
            support = cached_support

        teacher_log_probs_on_student_topk = torch.gather(
            selected_teacher_full_log_probs,
            dim=-1,
            index=student_topk_ids,
        )
        student_mass = student_topk_log_probs.exp().sum(dim=-1)
        teacher_mass = teacher_log_probs_on_student_topk.exp().sum(dim=-1)

        if loss_config.log_prob_min_clamp is not None:
            student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
            teacher_log_probs_on_student_topk = teacher_log_probs_on_student_topk.clamp_min(
                loss_config.log_prob_min_clamp
            )

        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_log_probs_on_student_topk = teacher_log_probs_on_student_topk - torch.logsumexp(
            teacher_log_probs_on_student_topk, dim=-1, keepdim=True
        )
        student_topk_probs = student_topk_log_probs.exp()
        token_reverse_kl = student_topk_probs * (student_topk_log_probs - teacher_log_probs_on_student_topk)
        selected_losses = token_reverse_kl.sum(dim=-1)

        token_losses = student_logits_flat.sum(dim=-1) * 0.0
        token_losses = token_losses.index_copy(0, predictor_positions, selected_losses)
        student_mass_out = torch.zeros_like(token_losses).index_copy(0, predictor_positions, student_mass)
        teacher_mass_out = torch.zeros_like(token_losses).index_copy(0, predictor_positions, teacher_mass)
        overlap_count_out = torch.zeros_like(token_losses).index_copy(
            0,
            predictor_positions,
            torch.full_like(selected_losses, float(topk)),
        )
        overlap_token_advantage = (-token_reverse_kl).mean(dim=-1)
        overlap_token_advantage_out = torch.zeros_like(token_losses).index_copy(
            0,
            predictor_positions,
            overlap_token_advantage,
        )
        return {
            "distillation_losses": token_losses.unsqueeze(0),
            "student_mass": student_mass_out.unsqueeze(0),
            "teacher_mass": teacher_mass_out.unsqueeze(0),
            "overlap_count": overlap_count_out.unsqueeze(0),
            "overlap_token_advantage": overlap_token_advantage_out.unsqueeze(0),
        }, collected_indices, support

    outputs, expected_indices, cached_support = _compute_on_student_support(
        teacher_full_log_probs,
        teacher_response_indices,
    )
    if (unprivileged_teacher_full_log_probs is None) != (unprivileged_teacher_response_indices is None):
        raise ValueError("Unprivileged teacher dense log probabilities and response indices must be provided together.")
    if unprivileged_teacher_full_log_probs is not None and unprivileged_teacher_response_indices is not None:
        unprivileged_outputs, _, _ = _compute_on_student_support(
            unprivileged_teacher_full_log_probs,
            unprivileged_teacher_response_indices,
            expected_response_indices=expected_indices,
            cached_support=cached_support,
        )
        outputs.update({f"unprivileged_{key}": value.detach() for key, value in unprivileged_outputs.items()})
    return outputs

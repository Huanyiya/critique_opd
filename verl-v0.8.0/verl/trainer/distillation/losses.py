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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
        use_full_vocab (bool): Whether the loss function uses full-vocabulary teacher log probabilities.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_full_vocab: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator, self.use_full_vocab]) != 1:
            raise ValueError(
                "Expected exactly one teacher-distribution representation, but got "
                f"{self.use_estimator=}, {self.use_topk=}, {self.use_full_vocab=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def _as_batch_list(value: Any, batch_size: int, default: Any = None) -> list[Any]:
    """Normalize TensorDict non-tensor values to a per-sample Python list."""
    if value is None:
        return [default for _ in range(batch_size)]
    if isinstance(value, torch.Tensor):
        if value.numel() == batch_size:
            return value.detach().cpu().tolist()
        if value.numel() == 1:
            scalar = value.detach().cpu().item()
            return [scalar for _ in range(batch_size)]
        return [default for _ in range(batch_size)]
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == batch_size:
            return value
        if len(value) == 1:
            return value * batch_size
        return [default for _ in range(batch_size)]
    return [value for _ in range(batch_size)]


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.detach().cpu().flatten()[0].item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _truthy(value: Any) -> bool:
    value = _scalar(value, False)
    return bool(value)


def _batch_metadata(data: TensorDict, batch_size: int) -> list[dict[str, Any]]:
    """Read ALFWorld OPD metadata from either TQ extra_fields or normal non-tensor fields."""
    extra_fields = tu.get(data, "extra_fields", None)
    extra_fields_list = _as_batch_list(extra_fields, batch_size, default={})
    if any(isinstance(item, dict) and item for item in extra_fields_list):
        return [item if isinstance(item, dict) else {} for item in extra_fields_list]

    metadata: list[dict[str, Any]] = [dict() for _ in range(batch_size)]
    for key in (
        "trajectory_uid",
        "task_uid",
        "opd_selected",
        "opd_step_index",
        "opd_error_step",
        "opd_skip_reason",
        "teacher_critique_parse_ok",
        "environment_reward",
        "trajectory_steps",
        "termination_reason",
    ):
        values = _as_batch_list(tu.get(data, key, None), batch_size, default=None)
        for idx, value in enumerate(values):
            metadata[idx][key] = value
    return metadata


def _trajectory_metric_mean(
    token_metric: torch.Tensor | None,
    response_mask_bool: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    scope: str,
) -> tuple[torch.Tensor | None, int]:
    """Aggregate token metrics by trajectory first, then average trajectories."""
    if token_metric is None:
        return None, 0
    if token_metric.shape != response_mask_bool.shape:
        raise ValueError(f"Token metric shape {token_metric.shape} does not match response mask {response_mask_bool.shape}.")

    groups: dict[str, list[torch.Tensor]] = {}
    for sample_idx, meta in enumerate(metadata):
        if not _truthy(meta.get("opd_selected")):
            continue
        step_index = _scalar(meta.get("opd_step_index"), None)
        error_step = _scalar(meta.get("opd_error_step"), None)
        if step_index is None or error_step is None:
            continue
        if scope == "error_action" and int(step_index) != int(error_step):
            continue
        sample_metric = token_metric[sample_idx]
        finite_mask = torch.isfinite(sample_metric)
        sample_mask = response_mask_bool[sample_idx] & finite_mask
        if not bool(sample_mask.any()):
            continue
        trajectory_uid = meta.get("trajectory_uid") or f"sample-{sample_idx}"
        groups.setdefault(str(trajectory_uid), []).append(sample_metric[sample_mask].detach())

    trajectory_means = [torch.cat(values).mean() for values in groups.values() if values]
    if not trajectory_means:
        return None, 0
    return torch.stack(trajectory_means).mean(), len(trajectory_means)


def _add_scoped_token_metrics(
    metrics: dict[str, Any],
    *,
    name: str,
    token_metric: torch.Tensor | None,
    response_mask_bool: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> None:
    for scope in ("prefix_to_error", "error_action"):
        value, _ = _trajectory_metric_mean(
            token_metric,
            response_mask_bool,
            metadata,
            scope=scope,
        )
        if value is not None:
            metrics[f"{name}_{scope}"] = value.detach().item()


def _opd_valid_trajectory_count(response_mask_bool: torch.Tensor, metadata: list[dict[str, Any]]) -> int:
    valid_uids: set[str] = set()
    for sample_idx, meta in enumerate(metadata):
        if not _truthy(meta.get("opd_selected")):
            continue
        if bool(response_mask_bool[sample_idx].any()):
            valid_uids.add(str(meta.get("trajectory_uid") or f"sample-{sample_idx}"))
    return len(valid_uids)


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute a distributional distillation loss in the logits processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_settings = distillation_config.distillation_loss.loss_settings
    if loss_settings.use_full_vocab:
        if config.strategy != "fsdp":
            raise NotImplementedError(
                "Full-vocabulary reverse KL currently requires the FSDP actor strategy, "
                f"but got {config.strategy=}."
            )
        import verl.trainer.distillation.fsdp.losses as fsdp_losses

        return fsdp_losses.compute_reverse_kl_full_vocab(
            student_logits=student_logits,
            teacher_full_log_probs=data["teacher_full_logprobs"],
            teacher_response_indices=data["teacher_response_indices"],
            data=data,
            config=distillation_config,
        )

    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if distillation_config.distillation_loss.loss_mode == "reverse_kl_topk":
                outputs = fsdp_losses.compute_reverse_kl_topk(
                    student_logits=student_logits,
                    student_topk_ids=data["student_topk_ids"],
                    teacher_topk_log_probs=data["teacher_topk_logprobs"],
                    teacher_topk_ids=data.get("teacher_topk_ids"),
                    teacher_own_topk_log_probs=data.get("teacher_own_topk_logprobs"),
                    teacher_response_indices=data["teacher_response_indices"],
                    unprivileged_teacher_topk_log_probs=data.get("unprivileged_teacher_topk_logprobs"),
                    unprivileged_teacher_topk_ids=data.get("unprivileged_teacher_topk_ids"),
                    unprivileged_teacher_own_topk_log_probs=data.get(
                        "unprivileged_teacher_own_topk_logprobs"
                    ),
                    unprivileged_teacher_response_indices=data.get("unprivileged_teacher_response_indices"),
                    data=data,
                    config=distillation_config,
                    data_format=data_format,
                )
                expected_shape = student_logits.shape[:2]
                for key, value in outputs.items():
                    assert value.shape == expected_shape, (
                        f"Expected shape {expected_shape}, but got {value.shape} for {key=}."
                    )
                return outputs
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            if distillation_config.distillation_loss.loss_mode == "reverse_kl_topk":
                raise NotImplementedError("reverse_kl_topk currently requires the FSDP or VeOmni actor strategy.")
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    response_mask = data["response_mask"]
    response_mask_values = response_mask.values() if response_mask.is_nested else response_mask
    if not response_mask_values.bool().any():
        # A critique-conditioned OPD batch can contain no failed trajectories.
        # Keep the zero connected to the current student forward pass so backward
        # remains valid, while avoiding empty-mask reductions in PPO/distillation metrics.
        zero_source = model_output.get("distillation_losses", model_output.get("log_probs"))
        zero_values = zero_source.values() if zero_source.is_nested else zero_source
        zero_loss = zero_values.sum() * 0.0
        return zero_loss, {
            "distillation/empty_opd_batch": 1.0,
            "distillation/loss": Metric(value=zero_loss.detach(), aggregation=AggregationType.SUM),
        }

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)
    policy_metrics["train/opd_loss"] = Metric(value=distill_loss.detach(), aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["forward_kl_topk", "reverse_kl_topk"], use_topk=True)
)  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return top-k distributional loss and metrics computed by the logits processor.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape
    metadata = _batch_metadata(data, batch_size=response_mask_bool.shape[0])

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    def _padded_optional(name: str) -> torch.Tensor | None:
        value = model_output.get(name)
        if value is None:
            return None
        return no_padding_2_padding(value, data)

    critique_js = _padded_optional("js_student_teacher")
    base_js = _padded_optional("unprivileged_js_student_teacher")
    critique_overlap = _padded_optional("topk_overlap_student_teacher")
    base_overlap = _padded_optional("unprivileged_topk_overlap_student_teacher")
    critique_token_logprob = _padded_optional("teacher_student_token_logprob")
    base_token_logprob = _padded_optional("unprivileged_teacher_student_token_logprob")
    critique_entropy = _padded_optional("teacher_entropy_topk_normalized")
    base_entropy = _padded_optional("unprivileged_teacher_entropy_topk_normalized")

    md_metrics: dict[str, Any] = {
        "train/opd_valid_token_count": Metric(
            AggregationType.SUM,
            response_mask_bool.sum().detach().to(dtype=torch.float32),
        ),
        "train/opd_valid_trajectory_count": Metric(
            AggregationType.SUM,
            torch.tensor(
                float(_opd_valid_trajectory_count(response_mask_bool, metadata)),
                device=response_mask_bool.device,
            ),
        ),
    }
    _add_scoped_token_metrics(
        md_metrics,
        name="distribution/js_student_base",
        token_metric=base_js,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="distribution/js_student_critique",
        token_metric=critique_js,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    if critique_js is not None and base_js is not None:
        _add_scoped_token_metrics(
            md_metrics,
            name="distribution/js_critique_effect",
            token_metric=critique_js - base_js,
            response_mask_bool=response_mask_bool,
            metadata=metadata,
        )
    _add_scoped_token_metrics(
        md_metrics,
        name="distribution/topk_overlap_student_base",
        token_metric=base_overlap,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="distribution/topk_overlap_student_critique",
        token_metric=critique_overlap,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    if critique_overlap is not None and base_overlap is not None:
        _add_scoped_token_metrics(
            md_metrics,
            name="distribution/topk_overlap_critique_effect",
            token_metric=critique_overlap - base_overlap,
            response_mask_bool=response_mask_bool,
            metadata=metadata,
        )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/student_token_logprob_base",
        token_metric=base_token_logprob,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/student_token_logprob_critique",
        token_metric=critique_token_logprob,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    if critique_token_logprob is not None and base_token_logprob is not None:
        _add_scoped_token_metrics(
            md_metrics,
            name="teacher/student_token_logprob_critique_minus_base",
            token_metric=critique_token_logprob - base_token_logprob,
            response_mask_bool=response_mask_bool,
            metadata=metadata,
        )
        _add_scoped_token_metrics(
            md_metrics,
            name="teacher/student_token_prob_critique_minus_base",
            token_metric=critique_token_logprob.exp() - base_token_logprob.exp(),
            response_mask_bool=response_mask_bool,
            metadata=metadata,
        )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/student_token_prob_base",
        token_metric=base_token_logprob.exp() if base_token_logprob is not None else None,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/student_token_prob_critique",
        token_metric=critique_token_logprob.exp() if critique_token_logprob is not None else None,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/entropy_topk_normalized_base",
        token_metric=base_entropy,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )
    _add_scoped_token_metrics(
        md_metrics,
        name="teacher/entropy_topk_normalized_critique",
        token_metric=critique_entropy,
        response_mask_bool=response_mask_bool,
        metadata=metadata,
    )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
        **md_metrics,
    }

    # This guards normalized reverse KL against tiny negative roundoff and
    # preserves the native forward-top-k handling of negative partial sums.
    distillation_losses = distillation_losses.clamp_min(0.0)

    unprivileged_losses = model_output.get("unprivileged_distillation_losses")
    if unprivileged_losses is not None:
        unprivileged_losses = no_padding_2_padding(unprivileged_losses, data).clamp_min(0.0)
        unprivileged_teacher_mass = no_padding_2_padding(model_output["unprivileged_teacher_mass"], data)
        assert unprivileged_losses.shape == unprivileged_teacher_mass.shape == response_mask_bool.shape
        privileged_valid = distillation_losses[response_mask_bool]
        unprivileged_valid = unprivileged_losses[response_mask_bool]
        unprivileged_teacher_mass_valid = unprivileged_teacher_mass[response_mask_bool]
        distillation_metrics.update(
            {
                "distillation/privileged_reverse_kl": privileged_valid.mean().item(),
                "distillation/unprivileged_reverse_kl": unprivileged_valid.mean().item(),
                "distillation/privileged_minus_unprivileged_reverse_kl": (
                    privileged_valid - unprivileged_valid
                ).mean().item(),
                "distillation/unprivileged_teacher_mass": unprivileged_teacher_mass_valid.mean().item(),
            }
        )

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["reverse_kl_full_vocab"], use_full_vocab=True)
)  # type: ignore[arg-type]
def compute_reverse_kl_full_vocab(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return selected-token OPD losses for the original student response tokens.

    The historical loss-mode name is kept for config compatibility, but the
    actor-side logits processor now gathers only the student-selected token at
    each trainable response position instead of reducing over the whole vocabulary.
    """
    del config, distillation_config
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    return distillation_losses, {}


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics

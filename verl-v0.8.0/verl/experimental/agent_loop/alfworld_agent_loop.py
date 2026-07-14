# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Text-only ALFWorld agent loop for native veRL on-policy distillation."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol
from uuid import uuid4

import yaml

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.teacher_prompt import ALFWorldCritiqueTeacherPromptBuilder, TeacherPromptBuilder
from verl.utils.chat_template import apply_chat_template
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", flags=re.IGNORECASE | re.DOTALL)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
TASK_MARKER = "Your task is to: "


@dataclass(frozen=True)
class ParsedAction:
    """Action extracted from a model turn."""

    action: str
    is_valid: bool
    format_valid: bool
    admissible: bool


@dataclass(frozen=True)
class ALFWorldTransition:
    """Normalized result from one text-only ALFWorld step."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class ALFWorldEnvironment(Protocol):
    """Minimal synchronous environment contract used by the agent loop."""

    def reset(self) -> tuple[str, dict[str, Any]]: ...

    def step(self, action: str) -> ALFWorldTransition: ...

    def close(self) -> None: ...


def parse_alfworld_action(model_text: str, admissible_actions: list[str]) -> ParsedAction:
    """Parse OPID-style ``<action>`` output without requiring a thinking block."""
    lowered = model_text.lower()
    match = ACTION_PATTERN.search(lowered)
    format_valid = match is not None and CHINESE_PATTERN.search(model_text) is None
    # OPID forwards a short suffix for malformed output instead of replacing it
    # with an oracle action. Preserve that behavior and let ALFWorld respond.
    action = match.group(1).strip() if match is not None else lowered[-30:].strip()

    admissible_lookup = {
        candidate.lower().strip(): candidate for candidate in admissible_actions if candidate != "help"
    }
    is_admissible = action in admissible_lookup
    if is_admissible:
        action = admissible_lookup[action]
    return ParsedAction(
        action=action,
        is_valid=format_valid and is_admissible,
        format_valid=format_valid,
        admissible=is_admissible,
    )


def extract_task_description(observation: str) -> str:
    """Extract the ALFWorld task description from its initial observation."""
    marker_index = observation.find(TASK_MARKER)
    if marker_index < 0:
        raise ValueError(f"ALFWorld initial observation does not contain {TASK_MARKER!r}.")
    return observation[marker_index + len(TASK_MARKER) :].strip()


def build_alfworld_prompt(
    *,
    task_description: str,
    observation: str,
    admissible_actions: list[str],
    history: list[dict[str, Any]],
    step_index: int,
    history_length: int,
) -> str:
    """Build one action prompt from task, current observation, and recent history."""
    recent_history = history[-history_length:] if history_length > 0 else []
    if recent_history:
        history_text = "\n\n".join(
            (
                f"Step {item['step_index'] + 1}\n"
                f"Action: {item['action']}\n"
                f"Observation: {item['observation']}"
            )
            for item in recent_history
        )
    else:
        history_text = "No previous actions."

    action_text = "\n".join(f"- {action}" for action in admissible_actions if action != "help")
    return (
        "You are an agent operating in the text-only ALFWorld environment.\n"
        f"Task: {task_description}\n"
        f"Current step: {step_index + 1}\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current observation: {observation}\n"
        f"Admissible actions:\n{action_text}\n\n"
        "Choose exactly one admissible action. Reply with only <action>your action</action>."
    )


def _first(value: Any) -> Any:
    """Unwrap the batch dimension returned by a batch-size-one ALFWorld env."""
    if isinstance(value, (list, tuple)):
        return value[0]
    if hasattr(value, "ndim") and getattr(value, "ndim") > 0:
        return value[0]
    return value


def _flatten_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: _first(value) for key, value in info.items()}


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    return value


class InstalledALFWorldEnvironment:
    """Adapter around the installed ``alfworld`` package's batch-size-one API."""

    def __init__(
        self,
        *,
        config_path: Optional[str],
        split: str,
        seed: int,
        num_games: Optional[int] = None,
    ):
        try:
            import alfworld
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise ImportError(
                "Text-only ALFWorld rollout requires the optional `alfworld` package. "
                "Install it and run `alfworld-download -f`."
            ) from exc

        config_path = self._resolve_config_path(config_path, Path(alfworld.__file__).resolve().parent)
        with config_path.open(encoding="utf-8") as config_file:
            config = _expand_environment_variables(yaml.safe_load(config_file))

        config.setdefault("env", {})["type"] = "AlfredTWEnv"
        config.setdefault("general", {})["use_cuda"] = False
        if num_games is not None:
            config.setdefault("dataset", {})["num_train_games"] = int(num_games)
            config["dataset"]["num_eval_games"] = int(num_games)

        base_env = get_environment("AlfredTWEnv")(config, train_eval=split)
        self._base_env = base_env
        self._env = base_env.init_env(batch_size=1)
        self._env.seed(seed)

    @staticmethod
    def _resolve_config_path(config_path: Optional[str], package_root: Path) -> Path:
        if config_path and str(config_path).lower() not in {"none", "null"}:
            resolved = Path(os.path.expandvars(os.path.expanduser(str(config_path)))).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"ALFWorld config file does not exist: {resolved}")
            return resolved

        candidates = (
            package_root / "configs" / "base_config.yaml",
            package_root.parent / "configs" / "base_config.yaml",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Could not locate ALFWorld's base_config.yaml. Set ALFWORLD_CONFIG_PATH to an official "
            "text-world ALFWorld config file."
        )

    def reset(self) -> tuple[str, dict[str, Any]]:
        observations, info = self._env.reset()
        return str(_first(observations)), _flatten_info(info)

    def step(self, action: str) -> ALFWorldTransition:
        observations, _scores, dones, info = self._env.step([action])
        flat_info = _flatten_info(info)
        # Match OPID's text-only environment reward: 10 for a solved task, 0 otherwise.
        reward = 10.0 * float(bool(flat_info.get("won", False)))
        return ALFWorldTransition(
            observation=str(_first(observations)),
            reward=reward,
            done=bool(_first(dones)),
            info=flat_info,
        )

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        base_close = getattr(self._base_env, "close", None)
        if callable(base_close):
            base_close()


class InstalledALFWorldEnvironmentFactory:
    """Construct independent ALFWorld instances for concurrent trajectories."""

    def __init__(self, config_path: Optional[str], num_games: Optional[int]):
        self.config_path = config_path
        self.num_games = num_games

    def __call__(self, *, split: str, seed: int) -> ALFWorldEnvironment:
        return InstalledALFWorldEnvironment(
            config_path=self.config_path,
            split=split,
            seed=seed,
            num_games=self.num_games,
        )


class ALFWorldAgentLoop(AgentLoopBase):
    """Collect one on-policy ALFWorld trajectory entirely from the student."""

    def __init__(
        self,
        *args,
        config_path: Optional[str] = None,
        train_split: str = "train",
        eval_split: str = "eval_in_distribution",
        seed: int = 0,
        num_games: Optional[int] = None,
        max_steps: int = 30,
        history_length: int = 5,
        max_action_tokens: int = 128,
        teacher_critique_max_tokens: int = 256,
        teacher_prompt_builder: Optional[TeacherPromptBuilder] = None,
        environment_factory: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_split = train_split
        self.eval_split = eval_split
        self.seed = int(seed)
        self.num_games = (
            int(num_games) if num_games is not None and str(num_games).lower() not in {"none", "null", ""} else None
        )
        self.max_steps = int(max_steps)
        self.history_length = int(history_length)
        self.max_action_tokens = int(max_action_tokens)
        self.teacher_critique_max_tokens = int(teacher_critique_max_tokens)
        self.response_length = self.rollout_config.response_length
        self.teacher_prompt_builder = teacher_prompt_builder or ALFWorldCritiqueTeacherPromptBuilder()
        self.environment_factory = environment_factory or InstalledALFWorldEnvironmentFactory(
            config_path=config_path,
            num_games=self.num_games,
        )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}.")
        if self.max_action_tokens <= 0:
            raise ValueError(f"max_action_tokens must be positive, got {self.max_action_tokens}.")
        if self.teacher_critique_max_tokens <= 0:
            raise ValueError(
                f"teacher_critique_max_tokens must be positive, got {self.teacher_critique_max_tokens}."
            )
        if self.response_length < 2:
            raise ValueError(f"rollout.response_length must be at least 2, got {self.response_length}.")

    def _resolve_split(self, extra_info: dict[str, Any]) -> str:
        requested = extra_info.get("alfworld_split") or extra_info.get("split")
        if requested in {"eval_in_distribution", "eval_out_of_distribution", "train"}:
            return requested
        if requested in {"test", "validation", "val"}:
            return self.eval_split
        return self.train_split

    async def _tokenize_teacher_messages(self, messages: list[dict[str, str]]) -> list[int]:
        tokenized = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return normalize_token_ids(tokenized)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        sample_index = int(kwargs.get("index", extra_info.get("index", 0)))
        split = self._resolve_split(extra_info)
        environment = self.environment_factory(split=split, seed=self.seed + sample_index)

        generate_seconds = 0.0
        num_preempted = 0
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        response_step_end_indices: list[int] = []
        logprobs_available = True
        history: list[dict[str, Any]] = []
        turn_rewards: list[float] = []
        server_extra_fields: dict[str, Any] = {}
        environment_reward = 0.0
        termination_reason = "max_steps"

        try:
            initial_observation, reset_info = await self.loop.run_in_executor(None, environment.reset)
            task_description = extract_task_description(initial_observation)
            admissible_actions = list(reset_info.get("admissible_commands", []))
            task_uid = str(
                reset_info.get("extra.gamefile")
                or reset_info.get("gamefile")
                or f"alfworld-{split}-{sample_index}"
            )
            trajectory_uid = uuid4().hex

            initial_prompt = build_alfworld_prompt(
                task_description=task_description,
                observation=initial_observation,
                admissible_actions=admissible_actions,
                history=history,
                step_index=0,
                history_length=self.history_length,
            )
            student_prompt_ids = await self.apply_chat_template([{"role": "user", "content": initial_prompt}])

            for step_index in range(self.max_steps):
                remaining_tokens = self.response_length - len(response_ids)
                if remaining_tokens <= 1:
                    termination_reason = "response_length"
                    break

                turn_sampling_params = dict(sampling_params)
                turn_sampling_params["max_tokens"] = min(self.max_action_tokens, remaining_tokens - 1)
                started = time.perf_counter()
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=student_prompt_ids + response_ids,
                    sampling_params=turn_sampling_params,
                )
                generate_seconds += time.perf_counter() - started
                num_preempted += output.num_preempted if output.num_preempted is not None else 0
                if not server_extra_fields:
                    server_extra_fields.update(output.extra_fields)
                elif output.extra_fields.get("max_global_steps") is not None:
                    server_extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]

                generated_ids = list(output.token_ids[: remaining_tokens - 1])
                if not generated_ids:
                    termination_reason = "empty_generation"
                    break
                response_ids.extend(generated_ids)
                response_mask.extend([1] * len(generated_ids))
                response_step_end_indices.append(len(response_ids))
                if output.log_probs is None:
                    logprobs_available = False
                elif logprobs_available:
                    response_logprobs.extend(output.log_probs[: len(generated_ids)])

                model_text = await self.loop.run_in_executor(
                    None, lambda ids=generated_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
                )
                parsed_action = parse_alfworld_action(model_text, admissible_actions)
                transition = await self.loop.run_in_executor(None, environment.step, parsed_action.action)
                environment_reward += transition.reward
                turn_rewards.append(transition.reward)

                history_item = {
                    "task_uid": task_uid,
                    "trajectory_uid": trajectory_uid,
                    "step_index": step_index,
                    "model_output": model_text,
                    "action": parsed_action.action,
                    "observation": transition.observation,
                    "reward": transition.reward,
                    "done": transition.done,
                    "won": bool(transition.info.get("won", False)),
                    "is_action_valid": parsed_action.is_valid,
                    "format_valid": parsed_action.format_valid,
                    "admissible": parsed_action.admissible,
                }
                history.append(history_item)
                admissible_actions = list(transition.info.get("admissible_commands", []))

                observation_prompt = build_alfworld_prompt(
                    task_description=task_description,
                    observation=transition.observation,
                    admissible_actions=admissible_actions,
                    history=history,
                    step_index=step_index + 1,
                    history_length=self.history_length,
                )
                observation_ids = await self.apply_chat_template(
                    [{"role": "user", "content": observation_prompt}],
                    remove_system_prompt=True,
                )
                observation_ids = observation_ids[: self.response_length - len(response_ids)]
                response_ids.extend(observation_ids)
                response_mask.extend([0] * len(observation_ids))
                if logprobs_available:
                    response_logprobs.extend([0.0] * len(observation_ids))

                if transition.done:
                    termination_reason = "success" if bool(transition.info.get("won", False)) else "failure"
                    break
                if len(response_ids) >= self.response_length:
                    termination_reason = "response_length"
                    break
            else:
                termination_reason = "max_steps"

            opd_eligible = environment_reward <= 0.0 and bool(history)
            if opd_eligible:
                critique_messages = self.teacher_prompt_builder.build_critique_messages(
                    task_description=task_description,
                    initial_observation=initial_observation,
                    trajectory_steps=history,
                    task_uid=task_uid,
                    trajectory_uid=trajectory_uid,
                )
                teacher_critique_prompt_ids = await self._tokenize_teacher_messages(critique_messages)
                teacher_prompt_template = self.teacher_prompt_builder.build_scoring_prompt_template(
                    task_description=task_description,
                    initial_observation=initial_observation,
                    trajectory_steps=history,
                    task_uid=task_uid,
                    trajectory_uid=trajectory_uid,
                )
            else:
                # A non-None marker selects the critique-conditioned OPD path in
                # AgentLoopWorker, which excludes successful trajectories without
                # invoking the teacher.
                teacher_critique_prompt_ids = []
                teacher_prompt_template = None

            fallback_error_step = next(
                (index for index, step in enumerate(history) if not step["is_action_valid"]),
                max(len(history) - 1, 0),
            )
        finally:
            await self.loop.run_in_executor(None, environment.close)

        metrics = AgentLoopMetrics(
            generate_sequences=generate_seconds,
            tool_calls=0.0,
            compute_score=0.0,
            num_preempted=num_preempted,
        )
        extra_fields = {
            **server_extra_fields,
            "turn_scores": turn_rewards,
            "tool_rewards": [],
            "task_uid": task_uid,
            "trajectory_uid": trajectory_uid,
            "step_indices": [item["step_index"] for item in history],
            "trajectory_steps": history,
            "termination_reason": termination_reason,
            "environment_reward": environment_reward,
            "is_action_valid": [item["is_action_valid"] for item in history],
            "opd_eligible": opd_eligible,
        }
        return AgentLoopOutput(
            prompt_ids=student_prompt_ids,
            teacher_critique_prompt_ids=teacher_critique_prompt_ids,
            teacher_prompt_template=teacher_prompt_template,
            response_step_end_indices=response_step_end_indices,
            fallback_error_step=fallback_error_step,
            opd_eligible=opd_eligible,
            teacher_critique_max_tokens=self.teacher_critique_max_tokens,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if logprobs_available else None,
            reward_score=environment_reward,
            num_turns=1 + 2 * len(history),
            metrics=metrics,
            extra_fields=extra_fields,
        )

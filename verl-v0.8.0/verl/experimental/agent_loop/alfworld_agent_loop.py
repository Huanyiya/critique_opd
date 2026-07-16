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

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol
from uuid import uuid4

import numpy as np
import ray
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
ALFWORLD_TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
ALFWORLD_SPLIT_DIRS = {
    "train": "train",
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
}
_GAME_POOL_CACHE: dict[tuple[str, str], list[str]] = {}
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


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
    """Parse OPID-style ``<think>``/``<action>`` output."""
    original_text = model_text
    lowered = model_text.lower()
    match = ACTION_PATTERN.search(lowered)
    has_thinking_block = "<think>" in original_text and "</think>" in original_text
    has_no_chinese = CHINESE_PATTERN.search(original_text) is None
    format_valid = match is not None and has_thinking_block and has_no_chinese
    action = match.group(1).strip().lower() if match is not None else lowered[-30:]
    is_admissible = action in {candidate.strip().lower() for candidate in admissible_actions}
    is_valid = format_valid and is_admissible
    return ParsedAction(
        action=action,
        is_valid=is_valid,
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
    """Build one action prompt using OPID's text-only ALFWorld format."""
    action_text = "\n ".join(f"'{action}'" for action in admissible_actions if action != "help")
    if step_index == 0 or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=observation,
            admissible_actions=action_text,
        )

    recent_history = history[-history_length:]
    valid_history_length = len(recent_history)
    start_index = len(history) - valid_history_length
    history_text = "\n".join(
        (
            f"[Observation {start_index + history_offset + 1}: "
            f"'{item['prompt_observation']}', Action {start_index + history_offset + 1}: '{item['action']}']"
        )
        for history_offset, item in enumerate(recent_history)
    )
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=valid_history_length,
        action_history=history_text,
        current_step=step_index + 1,
        current_observation=observation,
        admissible_actions=action_text,
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


def _resolve_alfworld_data_root(data_root: Optional[str]) -> Path:
    if data_root is None or str(data_root).lower() in {"", "none", "null"}:
        data_root = os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
    return Path(os.path.expandvars(os.path.expanduser(str(data_root)))).resolve()


def _resolve_alfworld_split_dir(data_root: Optional[str], split: str) -> Path:
    root = _resolve_alfworld_data_root(data_root)
    split_dir_name = ALFWORLD_SPLIT_DIRS[split]
    candidates = (
        root / "json_2.1.1" / split_dir_name,
        root / split_dir_name,
        root,
    )
    for candidate in candidates:
        if candidate.name == split_dir_name and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find ALFWorld {split!r} directory under {root}. "
        f"Expected one of: {', '.join(str(candidate) for candidate in candidates)}"
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _collect_alfworld_game_pool(data_root: Optional[str], split: str) -> list[str]:
    """Collect the full legal, solvable ALFWorld game pool for one split."""
    split_dir = _resolve_alfworld_split_dir(data_root, split)
    cache_key = (str(split_dir), split)
    cached = _GAME_POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    allowed_task_types = set(ALFWORLD_TASK_TYPES.values())
    game_files: list[str] = []
    for root, dirs, files in os.walk(split_dir):
        dirs.sort()
        files.sort()
        if "traj_data.json" not in files:
            continue

        task_dir = Path(root)
        task_dir_text = str(task_dir)
        if "movable" in task_dir_text or "Sliced" in task_dir_text:
            continue

        game_file = task_dir / "game.tw-pddl"
        if not game_file.exists():
            continue

        traj_data = _read_json_file(task_dir / "traj_data.json")
        if traj_data.get("task_type") not in allowed_task_types:
            continue

        game_data = _read_json_file(game_file)
        if not game_data.get("solvable", False):
            continue

        game_files.append(str(game_file.resolve()))

    if not game_files:
        raise ValueError(f"No legal solvable ALFWorld games found for split={split!r} under {split_dir}.")

    _GAME_POOL_CACHE[cache_key] = game_files
    logger.warning("Loaded %d ALFWorld games for split=%s from %s", len(game_files), split, split_dir)
    return game_files


def _apply_alfworld_data_paths(config: dict[str, Any], data_root: Optional[str]) -> None:
    """Point an official ALFWorld config at the requested data root."""
    if data_root is None or str(data_root).lower() in {"", "none", "null"}:
        return

    root = _resolve_alfworld_data_root(data_root)
    dataset_config = config.setdefault("dataset", {})
    dataset_config["data_path"] = str(_resolve_alfworld_split_dir(root, "train"))
    dataset_config["eval_id_data_path"] = str(_resolve_alfworld_split_dir(root, "eval_in_distribution"))
    dataset_config["eval_ood_data_path"] = str(_resolve_alfworld_split_dir(root, "eval_out_of_distribution"))


class InstalledALFWorldEnvironment:
    """Adapter around the installed ``alfworld`` package's batch-size-one API."""

    def __init__(
        self,
        *,
        config_path: Optional[str],
        split: str,
        seed: int,
        data_root: Optional[str] = None,
        num_games: Optional[int] = None,
        game_file: Optional[str] = None,
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
        _apply_alfworld_data_paths(config, data_root)

        game_file_path = None
        if game_file is not None and str(game_file).lower() not in {"", "none", "null"}:
            game_file_path = Path(os.path.expandvars(os.path.expanduser(str(game_file)))).resolve()
            if not game_file_path.is_file():
                raise FileNotFoundError(f"ALFWorld game file does not exist: {game_file_path}")

            dataset_config = config.setdefault("dataset", {})
            game_dir = str(game_file_path.parent)
            if split == "train":
                dataset_config["data_path"] = game_dir
            elif split == "eval_in_distribution":
                dataset_config["eval_id_data_path"] = game_dir
            elif split == "eval_out_of_distribution":
                dataset_config["eval_ood_data_path"] = game_dir
            dataset_config["num_train_games"] = 1
            dataset_config["num_eval_games"] = 1
        elif num_games is not None:
            config.setdefault("dataset", {})["num_train_games"] = int(num_games)
            config["dataset"]["num_eval_games"] = int(num_games)
        else:
            config.setdefault("dataset", {})["num_train_games"] = 0
            config["dataset"]["num_eval_games"] = 0

        base_env = get_environment("AlfredTWEnv")(config, train_eval=split)
        if game_file_path is not None:
            # The official ALFWorld wrapper has already collected games in the
            # constructor. Keep the row-to-task mapping explicit even if the
            # package changes its traversal order.
            base_env.game_files = [str(game_file_path)]
            base_env.num_games = 1
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

        repo_root = Path(__file__).resolve().parents[3]
        candidates = (
            repo_root / "examples" / "opd" / "alfworld" / "config_tw.yaml",
            package_root / "configs" / "base_config.yaml",
            package_root.parent / "configs" / "base_config.yaml",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Could not locate an ALFWorld config file. Set ALFWORLD_CONFIG_PATH to a valid text-world "
            "ALFWorld YAML config."
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


class ALFWorldEnvWorker:
    """Long-lived Ray worker holding one batch-size-one ALFWorld environment."""

    def __init__(
        self,
        *,
        config_path: Optional[str],
        data_root: Optional[str],
        split: str,
        seed: int,
        num_games: Optional[int],
        base_env: Any,
        worker_id: int,
        group_id: int,
        rollout_id: int,
        resume_reset_count: int = 0,
    ):
        self.config_path = config_path
        self.data_root = data_root
        self.split = split
        self.seed = int(seed)
        self.worker_id = int(worker_id)
        self.group_id = int(group_id)
        self.rollout_id = int(rollout_id)
        self.num_games = num_games
        self._base_env = base_env
        self._game_files = list(getattr(base_env, "game_files", []) or [])
        self.reset_count = int(resume_reset_count)
        self._next_game_file: Optional[str] = None
        self._env = None
        self._initialize_env()

    def _make_textworld_iterator(self, reset_count: int) -> tuple[Any, Optional[str], int]:
        """Build a TextWorld game iterator positioned at reset_count without reset-skipping.

        TextWorld's batch env does:
            rng = np.random.RandomState(seed)
            rng.shuffle(gamefiles)
            yield the first shuffled order once
            rng.shuffle(order) before each later cycle

        Reconstruct that exact iterator state by replaying only per-cycle
        permutations, not by calling env.reset()/env.skip() reset_count times.
        """
        if not self._game_files:
            return iter(()), None, 0

        order = list(self._game_files)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(order)

        cycle_index, offset = divmod(max(int(reset_count), 0), len(order))
        for _ in range(cycle_index):
            rng.shuffle(order)

        next_game_file = str(order[offset])

        def iterator():
            current_order = order
            for game_file in current_order[offset:]:
                yield game_file
            while True:
                rng.shuffle(current_order)
                for game_file in current_order:
                    yield game_file

        return iterator(), next_game_file, offset

    def _restore_game_iterator(self) -> None:
        if self._env is None:
            return
        iterator, next_game_file, offset = self._make_textworld_iterator(self.reset_count)
        if self._game_files:
            setattr(self._env, "gamefiles", list(self._game_files))
            setattr(self._env, "_gamefiles_iterator", iterator)
        self._next_game_file = next_game_file
        logger.info(
            "ALFWorld worker %s restored: reset_count=%d cursor=%d next_game=%s",
            self.worker_id,
            self.reset_count,
            offset,
            next_game_file,
        )

    def _initialize_env(self) -> None:
        if self._game_files:
            self._base_env.game_files = list(self._game_files)
            self._base_env.num_games = len(self._game_files)
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        self._env = self._base_env.init_env(batch_size=1)
        self._env.seed(self.seed)
        self._restore_game_iterator()

    def set_reset_count(self, reset_count: int) -> dict[str, Any]:
        self.reset_count = int(reset_count)
        self._restore_game_iterator()
        return {
            "alfworld_worker_id": self.worker_id,
            "alfworld_group_id": self.group_id,
            "alfworld_rollout_id": self.rollout_id,
            "alfworld_reset_count": self.reset_count,
            "alfworld_next_game_file": self._next_game_file,
        }

    def ready(self) -> dict[str, int]:
        """Report readiness after the persistent environment is initialized."""
        return {
            "worker_id": self.worker_id,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
        }

    def reset(self) -> tuple[str, dict[str, Any]]:
        reset_count = self.reset_count
        expected_game_file = self._next_game_file
        observations, info = self._env.reset()
        self.reset_count += 1
        if self._game_files:
            _, self._next_game_file, _ = self._make_textworld_iterator(self.reset_count)
        flat_info = _flatten_info(info)
        flat_info.setdefault("alfworld_worker_id", self.worker_id)
        flat_info.setdefault("alfworld_group_id", self.group_id)
        flat_info.setdefault("alfworld_rollout_id", self.rollout_id)
        flat_info.setdefault("alfworld_reset_count", reset_count)
        if expected_game_file is not None:
            flat_info.setdefault("alfworld_expected_gamefile", expected_game_file)
        return str(_first(observations)), flat_info

    def step(self, action: str) -> ALFWorldTransition:
        observations, _scores, dones, info = self._env.step([action])
        flat_info = _flatten_info(info)
        flat_info.setdefault("alfworld_worker_id", self.worker_id)
        flat_info.setdefault("alfworld_group_id", self.group_id)
        flat_info.setdefault("alfworld_rollout_id", self.rollout_id)
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


class ALFWorldPersistentEnvironment:
    """Synchronous handle used by one trajectory to talk to a persistent Ray worker."""

    def __init__(self, worker: ray.actor.ActorHandle):
        self.worker = worker

    def reset(self) -> tuple[str, dict[str, Any]]:
        return ray.get(self.worker.reset.remote())

    def step(self, action: str) -> ALFWorldTransition:
        return ray.get(self.worker.step.remote(action))

    def close(self) -> None:
        # Persistent workers are owned by ALFWorldEnvManager and reused across
        # many trajectories. A per-trajectory close would destroy the pool.
        return None


class ALFWorldEnvManager:
    """Centrally own the persistent train and validation ALFWorld actors."""

    def __init__(
        self,
        *,
        config_path: Optional[str],
        data_root: Optional[str],
        train_split: str,
        eval_split: str,
        seed: int,
        num_games: Optional[int],
        train_num_groups: int,
        train_group_size: int,
        val_num_groups: int,
        val_group_size: int,
        env_worker_num_cpus: float,
    ):
        self.config_path = config_path
        self.data_root = data_root
        self.train_split = train_split
        self.eval_split = eval_split
        self.seed = int(seed)
        self.num_games = num_games
        self.train_num_groups = int(train_num_groups)
        self.train_group_size = int(train_group_size)
        self.val_num_groups = int(val_num_groups)
        self.val_group_size = int(val_group_size)
        self.env_worker_num_cpus = float(env_worker_num_cpus)
        self._actor_cls = ray.remote(num_cpus=self.env_worker_num_cpus)(ALFWorldEnvWorker)
        self._actors: dict[tuple[str, int], ray.actor.ActorHandle] = {}
        self._base_envs: dict[tuple[str, str], Any] = {}
        self._ready_pools: set[str] = set()
        self._closed = False
        self.train_resume_reset_count = 0

        if self.train_num_groups <= 0 or self.train_group_size <= 0:
            raise ValueError(
                f"train_num_groups and train_group_size must be positive, got "
                f"{self.train_num_groups} and {self.train_group_size}."
            )
        if self.val_num_groups <= 0 or self.val_group_size <= 0:
            raise ValueError(
                f"val_num_groups and val_group_size must be positive, got "
                f"{self.val_num_groups} and {self.val_group_size}."
            )

        # Match OPID's make_envs(): one owner creates both pools once during
        # trainer initialization. Agent-loop workers only receive this manager's
        # Ray handle and never create environment actors themselves.
        self._ensure_pool(validate=False)
        self._ensure_pool(validate=True)

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> Path:
        import alfworld

        return InstalledALFWorldEnvironment._resolve_config_path(
            config_path,
            Path(alfworld.__file__).resolve().parent,
        )

    def _load_config(self) -> dict[str, Any]:
        config_path = self._resolve_config_path(self.config_path)
        with config_path.open(encoding="utf-8") as config_file:
            config = _expand_environment_variables(yaml.safe_load(config_file))
        config.setdefault("env", {})["type"] = "AlfredTWEnv"
        config.setdefault("general", {})["use_cuda"] = False
        _apply_alfworld_data_paths(config, self.data_root)
        dataset_config = config.setdefault("dataset", {})
        if self.num_games is None:
            dataset_config["num_train_games"] = 0
            dataset_config["num_eval_games"] = 0
        else:
            dataset_config["num_train_games"] = int(self.num_games)
            dataset_config["num_eval_games"] = int(self.num_games)
        return config

    def _build_base_env(self, split: str) -> Any:
        cache_key = (split, str(self.config_path))
        if cache_key in self._base_envs:
            return self._base_envs[cache_key]

        from alfworld.agents.environment import get_environment

        config = self._load_config()
        game_pool = _collect_alfworld_game_pool(self.data_root, split)
        base_env = get_environment("AlfredTWEnv")(config, train_eval=split)
        if self.num_games is None:
            base_env.game_files = list(game_pool)
            base_env.num_games = len(game_pool)
        self._base_envs[cache_key] = base_env
        return base_env

    def _pool_spec(self, validate: bool) -> tuple[str, str, int, int, int]:
        if validate:
            return "val", self.eval_split, self.seed + 1000, self.val_num_groups, self.val_group_size
        return "train", self.train_split, self.seed, self.train_num_groups, self.train_group_size

    def _get_or_create_actor(
        self,
        *,
        pool_key: str,
        split: str,
        seed_base: int,
        group_size: int,
        worker_id: int,
    ) -> ray.actor.ActorHandle:
        actor_key = (pool_key, int(worker_id))
        cached = self._actors.get(actor_key)
        if cached is not None:
            return cached

        group_id = int(worker_id) // int(group_size)
        rollout_id = int(worker_id) % int(group_size)
        worker_seed = int(seed_base) + group_id
        base_env = self._build_base_env(split)
        actor = self._actor_cls.remote(
            config_path=self.config_path,
            data_root=self.data_root,
            split=split,
            seed=worker_seed,
            num_games=self.num_games,
            base_env=base_env,
            worker_id=worker_id,
            group_id=group_id,
            rollout_id=rollout_id,
            resume_reset_count=self.train_resume_reset_count if pool_key == "train" else 0,
        )

        self._actors[actor_key] = actor
        return actor

    def set_train_reset_count(self, reset_count: int) -> None:
        self.train_resume_reset_count = int(reset_count)
        self._ensure_pool(validate=False)
        train_worker_count = self.train_num_groups * self.train_group_size
        ray.get(
            [
                self._get_or_create_actor(
                    pool_key="train",
                    split=self.train_split,
                    seed_base=self.seed,
                    group_size=self.train_group_size,
                    worker_id=worker_id,
                ).set_reset_count.remote(self.train_resume_reset_count)
                for worker_id in range(train_worker_count)
            ]
        )
        logger.warning(
            "ALFWorld train env pool cursor restored to reset_count=%d for %d workers",
            self.train_resume_reset_count,
            train_worker_count,
        )

    def _ensure_pool(self, validate: bool) -> None:
        pool_key, split, seed_base, num_groups, group_size = self._pool_spec(validate)
        if pool_key in self._ready_pools:
            return
        actors = []
        for worker_id in range(num_groups * group_size):
            actors.append(
                self._get_or_create_actor(
                    pool_key=pool_key,
                    split=split,
                    seed_base=seed_base,
                    group_size=group_size,
                    worker_id=worker_id,
                )
            )
        # Wait for all actor constructors once so initialization failures surface
        # in TaskRunner before model training starts.
        ray.get([actor.ready.remote() for actor in actors])
        self._ready_pools.add(pool_key)
        logger.warning(
            "ALFWorld persistent %s pool ready: num_groups=%d group_size=%d total_workers=%d",
            pool_key,
            num_groups,
            group_size,
            num_groups * group_size,
        )

    def ready(self) -> dict[str, int]:
        return {
            "train_workers": self.train_num_groups * self.train_group_size,
            "val_workers": self.val_num_groups * self.val_group_size,
        }

    def get_worker(self, *, validate: bool, sample_index: int, session_id: int) -> ray.actor.ActorHandle:
        """Return the centrally owned worker assigned to one trajectory."""
        pool_key, split, seed_base, num_groups, group_size = self._pool_spec(validate)
        self._ensure_pool(validate=validate)

        group_id = int(sample_index)
        rollout_id = int(session_id)
        if group_id < 0 or group_id >= num_groups:
            raise IndexError(f"ALFWorld group_id={group_id} out of range for {pool_key} num_groups={num_groups}.")
        if rollout_id < 0 or rollout_id >= group_size:
            raise IndexError(f"ALFWorld rollout_id={rollout_id} out of range for {pool_key} group_size={group_size}.")

        worker_id = group_id * group_size + rollout_id
        actor = self._get_or_create_actor(
            pool_key=pool_key,
            split=split,
            seed_base=seed_base,
            group_size=group_size,
            worker_id=worker_id,
        )
        return actor

    def close(self) -> None:
        if self._closed or not ray.is_initialized():
            return
        self._closed = True
        actors = list(self._actors.values())
        if actors:
            try:
                ray.get([actor.close.remote() for actor in actors], timeout=30)
            except Exception:
                logger.warning("Some ALFWorld workers failed to close cleanly; killing the pool", exc_info=True)
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.debug("Ignoring ALFWorld worker kill failure", exc_info=True)
        self._actors.clear()
        self._ready_pools.clear()
        for base_env in self._base_envs.values():
            close = getattr(base_env, "close", None)
            if callable(close):
                close()
        self._base_envs.clear()


def _config_get(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def create_alfworld_env_manager_actor_from_config(config: Any) -> Optional[ray.actor.ActorHandle]:
    """Create the single TaskRunner-owned OPID-style environment manager."""
    default_agent_loop = _config_get(config, "actor_rollout_ref.rollout.agent.default_agent_loop")
    if default_agent_loop != "alfworld_opd":
        return None

    seed = int(os.environ.get("ALFWORLD_SEED", "0"))
    num_games_env = os.environ.get("ALFWORLD_NUM_GAMES")
    num_games = (
        int(num_games_env)
        if num_games_env is not None and str(num_games_env).lower() not in {"", "none", "null"}
        else None
    )
    train_num_groups = int(_config_get(config, "data.train_batch_size", 1))
    val_num_groups = int(_config_get(config, "data.val_batch_size", train_num_groups) or train_num_groups)
    train_group_size = int(_config_get(config, "actor_rollout_ref.rollout.n", 1) or 1)
    val_group_size = int(_config_get(config, "actor_rollout_ref.rollout.val_kwargs.n", 1) or 1)
    manager_cls = ray.remote(num_cpus=0)(ALFWorldEnvManager)
    manager = manager_cls.remote(
        config_path=os.environ.get("ALFWORLD_CONFIG_PATH"),
        data_root=os.environ.get("ALFWORLD_DATA"),
        train_split=os.environ.get("ALFWORLD_TRAIN_SPLIT", "train"),
        eval_split=os.environ.get("ALFWORLD_EVAL_SPLIT", "eval_in_distribution"),
        seed=seed,
        num_games=num_games,
        train_num_groups=train_num_groups,
        train_group_size=train_group_size,
        val_num_groups=val_num_groups,
        val_group_size=val_group_size,
        env_worker_num_cpus=float(os.environ.get("ALFWORLD_ENV_WORKER_NUM_CPUS", "0.1")),
    )
    # Waiting on a method guarantees that the manager constructor has created
    # and initialized both the train and validation worker pools.
    pool_sizes = ray.get(manager.ready.remote())
    logger.warning(
        "Central ALFWorld environment manager ready: train_workers=%d val_workers=%d",
        pool_sizes["train_workers"],
        pool_sizes["val_workers"],
    )
    return manager


class InstalledALFWorldEnvironmentFactory:
    """Construct independent ALFWorld instances for concurrent trajectories."""

    def __init__(self, config_path: Optional[str], num_games: Optional[int], data_root: Optional[str]):
        self.config_path = config_path
        self.num_games = num_games
        self.data_root = data_root

    def __call__(
        self,
        *,
        split: str,
        seed: int,
        game_file: Optional[str] = None,
    ) -> ALFWorldEnvironment:
        return InstalledALFWorldEnvironment(
            config_path=self.config_path,
            data_root=self.data_root,
            split=split,
            seed=seed,
            num_games=self.num_games,
            game_file=game_file,
        )


class ALFWorldAgentLoop(AgentLoopBase):
    """Collect one on-policy ALFWorld trajectory entirely from the student."""

    def __init__(
        self,
        *args,
        config_path: Optional[str] = None,
        data_root: Optional[str] = None,
        train_split: str = "train",
        eval_split: str = "eval_in_distribution",
        seed: int = 0,
        num_games: Optional[int] = None,
        env_worker_num_cpus: float = 0.1,
        max_steps: int = 30,
        history_length: int = 5,
        max_action_tokens: int = 128,
        teacher_critique_max_tokens: int = 256,
        teacher_critique_min_confidence: float = 0.1,
        teacher_critique_reject_log_path: Optional[str] = None,
        teacher_prompt_builder: Optional[TeacherPromptBuilder] = None,
        environment_factory: Optional[Any] = None,
        env_manager: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_split = train_split
        self.eval_split = eval_split
        self.data_root = data_root
        self.seed = int(seed)
        self.num_games = (
            int(num_games) if num_games is not None and str(num_games).lower() not in {"none", "null", ""} else None
        )
        self.env_worker_num_cpus = float(env_worker_num_cpus)
        self.max_steps = int(max_steps)
        self.history_length = int(history_length)
        self.max_action_tokens = int(max_action_tokens)
        self.teacher_critique_max_tokens = int(teacher_critique_max_tokens)
        self.teacher_critique_min_confidence = float(teacher_critique_min_confidence)
        self.teacher_critique_reject_log_path = (
            str(teacher_critique_reject_log_path)
            if teacher_critique_reject_log_path is not None
            and str(teacher_critique_reject_log_path).lower() not in {"", "none", "null"}
            else None
        )
        # In this OPID-style loop, rollout.response_length is the per-step
        # student action width. The full trajectory is kept in metadata and
        # per-step arrays; it is not packed into one long response tensor.
        self.response_length = int(self.rollout_config.response_length)
        self.teacher_prompt_builder = teacher_prompt_builder or ALFWorldCritiqueTeacherPromptBuilder()
        self.environment_factory = environment_factory
        self.env_manager = env_manager
        if self.environment_factory is None and self.env_manager is None:
            raise ValueError(
                "ALFWorldAgentLoop requires the TaskRunner-owned env_manager; "
                "only tests may substitute environment_factory."
            )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}.")
        if self.max_action_tokens <= 0:
            raise ValueError(f"max_action_tokens must be positive, got {self.max_action_tokens}.")
        if self.teacher_critique_max_tokens <= 0:
            raise ValueError(
                f"teacher_critique_max_tokens must be positive, got {self.teacher_critique_max_tokens}."
            )
        if not 0.0 <= self.teacher_critique_min_confidence <= 1.0:
            raise ValueError(
                "teacher_critique_min_confidence must be in [0, 1], "
                f"got {self.teacher_critique_min_confidence}."
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
        rollout_started = time.perf_counter()
        extra_info = kwargs.get("extra_info", {}) or {}
        sample_index = int(kwargs.get("index", extra_info.get("index", 0)))
        split = self._resolve_split(extra_info)
        session_id = int(kwargs.get("session_id", extra_info.get("session_id", 0)) or 0)
        validate = bool(kwargs.get("validate", extra_info.get("validate", False)))
        if self.env_manager is not None:
            worker = await self.loop.run_in_executor(
                None,
                lambda: ray.get(
                    self.env_manager.get_worker.remote(
                        validate=validate,
                        sample_index=sample_index,
                        session_id=session_id,
                    )
                ),
            )
            environment = ALFWorldPersistentEnvironment(worker)
        else:
            environment = self.environment_factory(split=split, seed=self.seed + sample_index)

        generate_seconds = 0.0
        num_preempted = 0
        summary_response_ids: list[int] = []
        summary_response_mask: list[int] = []
        summary_response_logprobs: list[float] = []
        response_step_end_indices: list[int] = []
        opd_step_prompt_ids: list[list[int]] = []
        opd_step_prompt_texts: list[str] = []
        opd_step_response_ids: list[list[int]] = []
        opd_step_response_logprobs: list[list[float]] = []
        opd_step_student_topk_ids: list[list[list[int]]] = []
        total_action_tokens = 0
        logprobs_available = True
        student_topk_ids_available = True
        history: list[dict[str, Any]] = []
        turn_rewards: list[float] = []
        server_extra_fields: dict[str, Any] = {}
        environment_reward = 0.0
        termination_reason = "max_steps"

        try:
            initial_observation, reset_info = await self.loop.run_in_executor(None, environment.reset)
            task_description = extract_task_description(initial_observation)
            admissible_actions = list(reset_info.get("admissible_commands", []))
            initial_admissible_actions = list(admissible_actions)
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
            current_prompt_ids = student_prompt_ids
            current_prompt_text = initial_prompt
            current_observation = initial_observation

            for step_index in range(self.max_steps):
                turn_sampling_params = dict(sampling_params)
                turn_max_tokens = min(self.max_action_tokens, self.response_length)
                turn_sampling_params["max_tokens"] = turn_max_tokens
                started = time.perf_counter()
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=current_prompt_ids,
                    sampling_params=turn_sampling_params,
                )
                generate_seconds += time.perf_counter() - started
                num_preempted += output.num_preempted if output.num_preempted is not None else 0
                lightweight_extra_fields = {
                    key: value
                    for key, value in output.extra_fields.items()
                    if key not in {"sample_ids", "sample_logprobs", "sample_full_logprobs"}
                }
                if not server_extra_fields:
                    server_extra_fields.update(lightweight_extra_fields)
                elif lightweight_extra_fields.get("max_global_steps") is not None:
                    server_extra_fields["max_global_steps"] = lightweight_extra_fields["max_global_steps"]

                generated_ids = list(output.token_ids[:turn_max_tokens])
                if not generated_ids:
                    termination_reason = "empty_generation"
                    break
                opd_step_prompt_ids.append(list(current_prompt_ids))
                opd_step_prompt_texts.append(current_prompt_text)
                opd_step_response_ids.append(list(generated_ids))
                sample_topk_ids = output.extra_fields.get("sample_ids")
                if sample_topk_ids is None:
                    student_topk_ids_available = False
                elif student_topk_ids_available:
                    current_topk_ids = [list(row) for row in sample_topk_ids[: len(generated_ids)]]
                    if len(current_topk_ids) != len(generated_ids):
                        student_topk_ids_available = False
                    else:
                        opd_step_student_topk_ids.append(current_topk_ids)
                total_action_tokens += len(generated_ids)
                response_step_end_indices.append(total_action_tokens)
                current_logprobs = None
                if output.log_probs is None:
                    logprobs_available = False
                elif logprobs_available:
                    current_logprobs = list(output.log_probs[: len(generated_ids)])
                    opd_step_response_logprobs.append(current_logprobs)
                if not summary_response_ids:
                    summary_response_ids = list(generated_ids)
                    summary_response_mask = [1] * len(generated_ids)
                    if current_logprobs is not None:
                        summary_response_logprobs = list(current_logprobs)

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
                    "prompt_observation": current_observation,
                    "admissible_actions_before": list(admissible_actions),
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

                if transition.done:
                    termination_reason = "success" if bool(transition.info.get("won", False)) else "failure"
                    break

                current_observation = transition.observation
                observation_prompt = build_alfworld_prompt(
                    task_description=task_description,
                    observation=current_observation,
                    admissible_actions=admissible_actions,
                    history=history,
                    step_index=step_index + 1,
                    history_length=self.history_length,
                )
                current_prompt_ids = await self.apply_chat_template([{"role": "user", "content": observation_prompt}])
                current_prompt_text = observation_prompt
            else:
                termination_reason = "max_steps"

            opd_eligible = environment_reward <= 0.0 and bool(history)
            if opd_eligible:
                critique_messages = self.teacher_prompt_builder.build_critique_messages(
                    task_description=task_description,
                    initial_observation=initial_observation,
                    initial_admissible_actions=initial_admissible_actions,
                    trajectory_steps=history,
                    task_uid=task_uid,
                    trajectory_uid=trajectory_uid,
                )
                teacher_critique_prompt_ids = await self._tokenize_teacher_messages(critique_messages)
                teacher_critique_prompt_text = critique_messages[-1]["content"] if critique_messages else None
                teacher_prompt_template = self.teacher_prompt_builder.build_scoring_prompt_template(
                    task_description=task_description,
                    initial_observation=initial_observation,
                    initial_admissible_actions=initial_admissible_actions,
                    trajectory_steps=history,
                    task_uid=task_uid,
                    trajectory_uid=trajectory_uid,
                )
            else:
                # A non-None marker selects the critique-conditioned OPD path in
                # AgentLoopWorker, which excludes successful trajectories without
                # invoking the teacher.
                teacher_critique_prompt_ids = []
                teacher_critique_prompt_text = None
                teacher_prompt_template = None

            fallback_error_step = next(
                (index for index, step in enumerate(history) if not step["is_action_valid"]),
                max(len(history) - 1, 0),
            )
        finally:
            # Persistent ALFWorld workers are owned by ALFWorldEnvManager and
            # reused across steps. They are closed/killed by the manager, not by
            # an individual trajectory.
            pass

        metrics = AgentLoopMetrics(
            generate_sequences=generate_seconds,
            tool_calls=0.0,
            compute_score=0.0,
            num_preempted=num_preempted,
        )
        if not summary_response_ids:
            fallback_token_id = self.tokenizer.eos_token_id
            if fallback_token_id is None:
                fallback_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            summary_response_ids = [int(fallback_token_id)]
            summary_response_mask = [0]
            if logprobs_available:
                summary_response_logprobs = [0.0]
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
            "student_rollout_sec": time.perf_counter() - rollout_started,
            "teacher_critique_prompt_text": teacher_critique_prompt_text,
        }
        return AgentLoopOutput(
            prompt_ids=student_prompt_ids,
            teacher_critique_prompt_ids=teacher_critique_prompt_ids,
            teacher_prompt_template=teacher_prompt_template,
            response_step_end_indices=response_step_end_indices,
            fallback_error_step=fallback_error_step,
            opd_eligible=opd_eligible,
            teacher_critique_max_tokens=self.teacher_critique_max_tokens,
            teacher_critique_min_confidence=self.teacher_critique_min_confidence,
            teacher_critique_reject_log_path=self.teacher_critique_reject_log_path,
            opd_step_prompt_ids=opd_step_prompt_ids,
            opd_step_prompt_texts=opd_step_prompt_texts,
            opd_step_response_ids=opd_step_response_ids,
            opd_step_response_logprobs=opd_step_response_logprobs
            if logprobs_available and len(opd_step_response_logprobs) == len(opd_step_response_ids)
            else None,
            opd_step_student_topk_ids=opd_step_student_topk_ids
            if student_topk_ids_available and len(opd_step_student_topk_ids) == len(opd_step_response_ids)
            else None,
            response_ids=summary_response_ids,
            response_mask=summary_response_mask,
            response_logprobs=summary_response_logprobs
            if logprobs_available and len(summary_response_logprobs) == len(summary_response_ids)
            else None,
            reward_score=environment_reward,
            num_turns=len(history),
            metrics=metrics,
            extra_fields=extra_fields,
        )

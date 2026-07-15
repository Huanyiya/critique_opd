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

import asyncio
from typing import Any, Optional

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.alfworld_agent_loop import (
    ALFWorldAgentLoop,
    ALFWorldTransition,
    parse_alfworld_action,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen_contents: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del tools, add_generation_prompt, tokenize
        assert kwargs.get("enable_thinking") is False
        contents = [message.get("content", "") for message in messages]
        if all(not content for content in contents):
            return [1]
        content = str(contents[-1])
        self.seen_contents.append(content)
        if "diagnosing a failed text-only ALFWorld trajectory" in content:
            return [301, 302]
        if "Your current observation is: You are in a kitchen." in content:
            return [101, 102]
        if "You are now at step 2" in content:
            return [201, 202]
        if "You are now at step 3" in content:
            return [203, 204]
        raise AssertionError(f"Unexpected prompt: {content}")

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if ids == [11, 12]:
            return "<think>I should inspect the room.</think><action>look</action>"
        if ids == [21, 22]:
            return "<think>The apple is the target object.</think><action>take apple</action>"
        return "malformed"


class _FakeServerManager:
    def __init__(self):
        self.calls: list[list[int]] = []
        self.outputs = ([11, 12], [21, 22])

    async def generate(self, *, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs) -> TokenOutput:
        del sampling_params, kwargs
        self.calls.append(list(prompt_ids))
        token_ids = list(self.outputs[len(self.calls) - 1])
        return TokenOutput(token_ids=token_ids, log_probs=[-0.1] * len(token_ids), num_preempted=0)


class _ImmediateEventLoop:
    """Execute blocking fakes inline so the test does not depend on a host thread pool."""

    def run_in_executor(self, executor, function, *args):
        del executor
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class _FakeEnvironment:
    def __init__(self, *, solve_on_step: Optional[int]):
        self.solve_on_step = solve_on_step
        self.step_count = 0
        self.actions: list[str] = []
        self.closed = False

    def reset(self):
        return (
            "You are in a kitchen. Your task is to: put the apple on the table",
            {"admissible_commands": ["look", "take apple"], "extra.gamefile": "/games/task-1/game.tw-pddl"},
        )

    def step(self, action: str) -> ALFWorldTransition:
        self.actions.append(action)
        self.step_count += 1
        won = self.solve_on_step == self.step_count
        return ALFWorldTransition(
            observation=f"observation {self.step_count}",
            reward=10.0 if won else 0.0,
            done=won,
            info={"won": won, "admissible_commands": ["look", "take apple"]},
        )

    def close(self):
        self.closed = True


def _make_loop(
    environment: _FakeEnvironment, *, max_steps: int
) -> tuple[ALFWorldAgentLoop, _FakeServerManager, _FakeTokenizer]:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 32, "response_length": 32},
            },
            "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    server = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    loop = ALFWorldAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        environment_factory=lambda **kwargs: environment,
        max_steps=max_steps,
        history_length=2,
        max_action_tokens=4,
    )
    loop.loop = _ImmediateEventLoop()
    return loop, server, tokenizer


def test_invalid_action_is_forwarded_without_oracle_replacement():
    parsed = parse_alfworld_action("answer without tags", ["look", "take apple"])
    assert parsed.action == "answer without tags"
    assert not parsed.format_valid
    assert not parsed.admissible
    assert not parsed.is_valid


def test_alfworld_rollout_preserves_response_tokens_and_masks_observations():
    environment = _FakeEnvironment(solve_on_step=2)

    async def run_scenario():
        loop, server, _tokenizer = _make_loop(environment, max_steps=4)
        output = await loop.run(
            sampling_params={"temperature": 1.0}, index=0, extra_info={"split": "train"}
        )
        return output, server

    output, server = asyncio.run(run_scenario())

    assert output.prompt_ids == [101, 102]
    assert output.opd_eligible is False
    assert output.teacher_critique_prompt_ids == []
    assert output.response_step_end_indices == [2, 6]
    assert output.response_ids == [11, 12, 201, 202, 21, 22]
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert output.response_logprobs == [-0.1, -0.1, 0.0, 0.0, -0.1, -0.1]
    # OPID-style rollout regenerates from the current step prompt, not accumulated trajectory tokens.
    assert server.calls[0] == [101, 102]
    assert server.calls[1] == [201, 202]
    assert environment.actions == ["look", "take apple"]
    assert environment.closed

    assert output.reward_score == 10.0
    assert output.extra_fields["termination_reason"] == "success"
    assert output.extra_fields["task_uid"] == "/games/task-1/game.tw-pddl"
    assert output.extra_fields["step_indices"] == [0, 1]
    assert len({step["trajectory_uid"] for step in output.extra_fields["trajectory_steps"]}) == 1
    assert all(output.extra_fields["is_action_valid"])


def test_alfworld_rollout_terminates_at_max_steps():
    environment = _FakeEnvironment(solve_on_step=None)

    async def run_scenario():
        loop, _server, tokenizer = _make_loop(environment, max_steps=1)
        output = await loop.run(sampling_params={}, index=1, extra_info={"split": "train"})
        return output, tokenizer

    output, tokenizer = asyncio.run(run_scenario())

    assert output.extra_fields["termination_reason"] == "max_steps"
    assert output.extra_fields["step_indices"] == [0]
    assert output.reward_score == 0.0
    assert output.response_mask == [1, 1, 0, 0]
    assert output.opd_eligible is True
    assert output.teacher_critique_prompt_ids == [301, 302]
    assert output.response_step_end_indices == [2]
    assert output.teacher_prompt_template.count("{error_step}") == 1
    assert output.teacher_prompt_template.count("{reason}") == 1
    assert output.teacher_prompt_template.count("{better_decision}") == 1
    assert "Task:\nput the apple on the table" in output.teacher_prompt_template
    critique_prompt = tokenizer.seen_contents[-1]
    assert "Task: put the apple on the table" in critique_prompt
    assert "Student model output: <think>I should inspect the room.</think><action>look</action>" in critique_prompt
    assert "Executed action: look" in critique_prompt
    assert "Environment observation: observation 1" in critique_prompt
    assert environment.closed

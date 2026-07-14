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
"""Privileged teacher-prompt interfaces for on-policy distillation."""

import re
from abc import ABC, abstractmethod
from typing import Any


ERROR_STEP_PATTERN = re.compile(r"<error_step>\s*(\d+)\s*</error_step>", flags=re.IGNORECASE)


def parse_critique_error_step(critique: str, *, num_steps: int, fallback_step: int) -> int:
    """Return a zero-based error step, falling back when teacher output is malformed."""
    if num_steps <= 0:
        raise ValueError("Cannot select an error step from an empty trajectory.")
    match = ERROR_STEP_PATTERN.search(critique)
    if match is not None:
        parsed_step = int(match.group(1)) - 1
        if 0 <= parsed_step < num_steps:
            return parsed_step
    return min(max(fallback_step, 0), num_steps - 1)


class TeacherPromptBuilder(ABC):
    """Build the two teacher prompts used by critique-conditioned OPD.

    The first prompt asks the teacher to locate the earliest error in a failed
    student trajectory and produce privileged critique ``c``. The second is a
    template for the prompt under which the teacher scores the original student
    response token IDs. Only ``c`` is substituted into that template; the student
    response is appended later without decoding or re-tokenizing it.
    """

    @abstractmethod
    def build_critique_messages(self, **context: Any) -> list[dict[str, str]]:
        """Return chat messages containing the task and full failed trajectory."""

    @abstractmethod
    def build_scoring_prompt_template(self, **context: Any) -> str:
        """Return a text template containing exactly one ``{critique}`` field."""


def format_alfworld_trajectory(trajectory_steps: list[dict[str, Any]]) -> str:
    """Format an ALFWorld trajectory for teacher diagnosis."""
    return "\n\n".join(
        (
            f"Step {step['step_index'] + 1}\n"
            f"Student model output: {step['model_output']}\n"
            f"Executed action: {step['action']}\n"
            f"Environment observation: {step['observation']}\n"
            f"Action accepted by environment: {step['admissible']}\n"
            f"Task solved after this step: {step['won']}"
        )
        for step in trajectory_steps
    )


class ALFWorldCritiqueTeacherPromptBuilder(TeacherPromptBuilder):
    """Construct structured critique and privileged scoring prompts for ALFWorld."""

    def build_critique_messages(
        self,
        *,
        task_description: str,
        initial_observation: str,
        trajectory_steps: list[dict[str, Any]],
        **context: Any,
    ) -> list[dict[str, str]]:
        del context
        trajectory = format_alfworld_trajectory(trajectory_steps)
        content = (
            "You are diagnosing a failed text-only ALFWorld trajectory.\n"
            f"Task: {task_description}\n\n"
            f"Initial environment observation: {initial_observation}\n\n"
            f"Failed student trajectory:\n{trajectory}\n\n"
            "Find the earliest student step that made the trajectory incorrect or prevented task completion. "
            "Use one-based step numbering and return exactly this structure:\n"
            "<error_step>N</error_step>\n"
            "<critique>A concise explanation of the error and the better decision at that step.</critique>"
        )
        return [{"role": "user", "content": content}]

    def build_scoring_prompt_template(self, *, task_description: str, **context: Any) -> str:
        del context
        return (
            "You are acting in the text-only ALFWorld environment.\n"
            f"Task: {task_description}\n\n"
            "Privileged diagnosis of a failed attempt:\n{critique}\n\n"
            "Using this diagnosis as private guidance, produce the ALFWorld interaction trajectory."
        )

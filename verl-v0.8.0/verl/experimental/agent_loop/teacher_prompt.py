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
from dataclasses import dataclass
from typing import Any


ERROR_STEP_PATTERN = re.compile(r"<error_step>\s*(\d+)\s*</error_step>", flags=re.IGNORECASE)
REASON_PATTERN = re.compile(r"<reason>\s*(.*?)\s*</reason>", flags=re.IGNORECASE | re.DOTALL)
BETTER_DECISION_PATTERN = re.compile(
    r"<better_decision>\s*(.*?)\s*</better_decision>", flags=re.IGNORECASE | re.DOTALL
)
CONFIDENCE_PATTERN = re.compile(
    r"<confidence>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*</confidence>",
    flags=re.IGNORECASE,
)
TURN_ANCHORS = (
    "Now it's your turn to",
    "Now it's your turn",
)


@dataclass(frozen=True)
class TeacherCritiqueFeedback:
    """Parsed structured hindsight feedback from the teacher."""

    error_step: int
    """Zero-based index of the earliest critical error."""
    reason: str
    """Why the selected step is the earliest critical error."""
    better_decision: str
    """A better action or decision at the selected step."""
    confidence: float
    """Teacher confidence that ``error_step`` is the earliest critical error, in [0, 1]."""
    parse_ok: bool
    """Whether all required feedback fields were present and valid."""
    parse_errors: tuple[str, ...]
    """Human-readable parse failures; empty when ``parse_ok`` is true."""


def _parse_required_tagged_text(pattern: re.Pattern[str], critique: str, field_name: str) -> tuple[str, str | None]:
    match = pattern.search(critique)
    if match is None:
        return "", f"missing <{field_name}>"
    value = match.group(1).strip()
    if not value:
        return "", f"empty <{field_name}>"
    return value, None


def parse_critique_feedback(
    critique: str,
    *,
    num_steps: int,
    fallback_step: int,
) -> TeacherCritiqueFeedback:
    """Parse structured teacher feedback, using safe fallbacks for malformed fields."""
    if num_steps <= 0:
        raise ValueError("Cannot select an error step from an empty trajectory.")

    parse_errors: list[str] = []
    error_step = min(max(fallback_step, 0), num_steps - 1)
    match = ERROR_STEP_PATTERN.search(critique)
    if match is None:
        parse_errors.append("missing <error_step>")
    else:
        parsed_step = int(match.group(1)) - 1
        if 0 <= parsed_step < num_steps:
            error_step = parsed_step
        else:
            parse_errors.append(f"<error_step> {parsed_step + 1} out of range 1..{num_steps}")

    reason, reason_error = _parse_required_tagged_text(REASON_PATTERN, critique, "reason")
    if reason_error is not None:
        parse_errors.append(reason_error)
        reason = "The failed attempt did not make the correct progress toward the task."

    better_decision, better_decision_error = _parse_required_tagged_text(
        BETTER_DECISION_PATTERN,
        critique,
        "better_decision",
    )
    if better_decision_error is not None:
        parse_errors.append(better_decision_error)
        better_decision = "Choose a valid action that makes direct progress toward the task."

    confidence = 0.0
    confidence_match = CONFIDENCE_PATTERN.search(critique)
    if confidence_match is None:
        parse_errors.append("missing or malformed <confidence>")
    else:
        confidence = float(confidence_match.group(1))
        if not 0.0 <= confidence <= 1.0:
            parse_errors.append(f"<confidence> {confidence} out of range [0, 1]")
            confidence = min(max(confidence, 0.0), 1.0)

    return TeacherCritiqueFeedback(
        error_step=error_step,
        reason=reason,
        better_decision=better_decision,
        confidence=confidence,
        parse_ok=not parse_errors,
        parse_errors=tuple(parse_errors),
    )


def parse_critique_error_step(critique: str, *, num_steps: int, fallback_step: int) -> int:
    """Return a zero-based error step, falling back when teacher output is malformed."""
    return parse_critique_feedback(
        critique,
        num_steps=num_steps,
        fallback_step=fallback_step,
    ).error_step


def insert_teacher_feedback_into_current_prompt(*, current_prompt: str, teacher_feedback: str) -> str:
    """Insert teacher-only feedback into the same current-step prompt shape used for rollout."""
    prompt_text = str(current_prompt).strip()
    feedback_text = str(teacher_feedback).strip()
    if not feedback_text:
        return prompt_text
    if not prompt_text:
        return feedback_text

    for anchor in TURN_ANCHORS:
        anchor_index = prompt_text.find(anchor)
        if anchor_index < 0:
            continue
        prefix = prompt_text[:anchor_index].rstrip()
        suffix = prompt_text[anchor_index:].lstrip()
        if prefix and suffix:
            return f"{prefix}\n\n{feedback_text}\n\n{suffix}"
        if prefix:
            return f"{prefix}\n\n{feedback_text}"
        return f"{feedback_text}\n\n{suffix}"

    return f"{prompt_text}\n\n{feedback_text}"


class TeacherPromptBuilder(ABC):
    """Build the two teacher prompts used by critique-conditioned OPD.

    The first prompt asks the teacher to locate the earliest error in a failed
    student trajectory and produce privileged critique ``c``. The second is a
    template for the prompt under which the teacher scores the original student
    response token IDs. The parsed error step, reason, better decision, and
    confidence are substituted into that template; the student response is appended later
    without decoding or re-tokenizing it.
    """

    @abstractmethod
    def build_critique_messages(self, **context: Any) -> list[dict[str, str]]:
        """Return chat messages containing the task and full failed trajectory."""

    @abstractmethod
    def build_scoring_prompt_template(self, **context: Any) -> str:
        """Return a template with feedback fields and an optional ``current_prompt`` field."""


def format_alfworld_trajectory(trajectory_steps: list[dict[str, Any]]) -> str:
    """Format an ALFWorld trajectory for teacher diagnosis."""
    formatted_steps = []
    for step in trajectory_steps:
        step_number = step["step_index"] + 1
        admissible_actions_before = step.get("admissible_actions_before", [])
        formatted_steps.append(
            f"Step {step_number}\n"
            f"Environment observation before Action {step_number}:\n"
            f"{step['prompt_observation']}\n\n"
            f"Admissible actions before Action {step_number}:\n"
            f"{format_admissible_actions(admissible_actions_before)}\n\n"
            f"Student action {step_number}:\n"
            f"{step['action']}\n\n"
            f"Action output format valid: {step['format_valid']}\n"
        )
    return "\n\n".join(formatted_steps)


def format_admissible_actions(admissible_actions: list[str]) -> str:
    """Format an ALFWorld admissible-action list for a teacher prompt."""
    actions = [action for action in admissible_actions if action != "help"]
    if not actions:
        return "(none provided)"
    return "\n".join(f"- {action}" for action in actions)


class ALFWorldCritiqueTeacherPromptBuilder(TeacherPromptBuilder):
    """Construct structured critique and privileged scoring prompts for ALFWorld."""

    def build_critique_messages(
        self,
        *,
        task_description: str,
        initial_observation: str,
        initial_admissible_actions: list[str],
        trajectory_steps: list[dict[str, Any]],
        **context: Any,
    ) -> list[dict[str, str]]:
        del context
        trajectory = format_alfworld_trajectory(trajectory_steps)
        content = (
            "You are diagnosing a failed text-only ALFWorld trajectory.\n"
            f"Task: {task_description}\n\n"
            f"Initial environment observation: {initial_observation}\n\n"
            "Initial admissible actions:\n"
            f"{format_admissible_actions(initial_admissible_actions)}\n\n"
            f"Failed student trajectory:\n{trajectory}\n\n"
            "Find the earliest student step that made the trajectory incorrect or prevented task completion. "
            "Use one-based step numbering and return exactly this structure:\n"
            "<error_step>N</error_step>\n"
            "<reason>A concise explanation of why this is the earliest critical error.</reason>\n"
            "<better_decision>The better action or decision at that step.</better_decision>\n"
            "<confidence>A number from 0.0 to 1.0 indicating how confident you are that this is the earliest "
            "critical error.</confidence>"
        )
        return [{"role": "user", "content": content}]

    def build_scoring_prompt_template(
        self,
        *,
        task_description: str,
        initial_observation: str,
        initial_admissible_actions: list[str],
        **context: Any,
    ) -> str:
        del task_description, initial_observation, initial_admissible_actions, context
        return (
            "Private hindsight feedback from a failed attempt:\n\n"
            "Earliest critical error: step {error_step}.\n"
            "Reason: {reason}\n"
            "Suggested action: {better_decision}\n"
            "Confidence: {confidence}\n\n"
            "Use this feedback silently when choosing actions.\n"
            "Do not mention the feedback."
        )

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
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.teacher_prompt import parse_critique_error_step
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    align_teacher_outputs_to_student_response,
    select_teacher_full_vocab_response_rows,
)
from verl.trainer.distillation.fsdp.losses import compute_reverse_kl_full_vocab
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.workers.config import DistillationLossConfig, DistillationTeacherModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs, extract_sample_logprobs


def test_align_teacher_outputs_by_response_position_when_prompt_lengths_differ():
    teacher_ids = torch.arange(12, dtype=torch.int32).reshape(6, 2)
    teacher_logprobs = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10

    aligned_ids, aligned_logprobs = align_teacher_outputs_to_student_response(
        teacher_ids,
        teacher_logprobs,
        teacher_prompt_length=3,
        student_prompt_length=5,
        response_length=3,
        pad_token_id=99,
    )

    assert aligned_ids.shape == aligned_logprobs.shape == (8, 2)
    # Teacher rows 2:5 predict the three response tokens. Student rows 4:7
    # predict those same response positions despite the longer student prompt.
    torch.testing.assert_close(aligned_ids[4:7], teacher_ids[2:5])
    torch.testing.assert_close(aligned_logprobs[4:7], teacher_logprobs[2:5])
    assert torch.all(aligned_ids[:4] == 99)
    assert torch.all(aligned_ids[7:] == 99)
    assert torch.count_nonzero(aligned_logprobs[:4]) == 0
    assert torch.count_nonzero(aligned_logprobs[7:]) == 0


def test_equal_prompt_lengths_preserve_native_teacher_outputs():
    teacher_ids = torch.tensor([[1], [2], [3], [0]], dtype=torch.int32)
    teacher_logprobs = torch.tensor([[-0.1], [-0.2], [-0.3], [0.0]])

    aligned_ids, aligned_logprobs = align_teacher_outputs_to_student_response(
        teacher_ids,
        teacher_logprobs,
        teacher_prompt_length=2,
        student_prompt_length=2,
        response_length=2,
        pad_token_id=0,
    )

    assert aligned_ids is teacher_ids
    assert aligned_logprobs is teacher_logprobs


def test_full_vocab_rows_use_response_positions_not_teacher_prompt_positions():
    teacher_full_logprobs = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)

    response_indices, selected = select_teacher_full_vocab_response_rows(
        teacher_full_logprobs,
        teacher_prompt_length=3,
        response_mask=[1, 0, 1, 0, 1],
    )

    torch.testing.assert_close(response_indices, torch.tensor([0, 2, 4]))
    # Rows 2, 4, and 6 predict response tokens 0, 2, and 4. The
    # response-local indices remain valid for any student prompt length.
    torch.testing.assert_close(selected, teacher_full_logprobs[[2, 4, 6]])


def test_full_vocab_reverse_kl_uses_complete_student_distribution_and_response_indices():
    vocab_size = 3
    student_logits = torch.zeros((1, 9, vocab_size), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        student_logits[0, 1] = torch.tensor([1.0, 0.0, -1.0])
        student_logits[0, 3] = torch.tensor([-0.5, 0.25, 0.75])

    teacher_rows = torch.nested.as_nested_tensor(
        [
            torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])),
            torch.empty((0, vocab_size)),
        ],
        layout=torch.jagged,
    )
    response_indices = torch.nested.as_nested_tensor(
        [torch.tensor([0, 2]), torch.empty((0,), dtype=torch.int64)], layout=torch.jagged
    )
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [torch.arange(5), torch.arange(4)], layout=torch.jagged
            ),
            "prompts": torch.nested.as_nested_tensor(
                [torch.arange(2), torch.arange(2)], layout=torch.jagged
            ),
            "teacher_full_logprobs": teacher_rows,
            "teacher_response_indices": response_indices,
        },
        batch_size=[2],
    )
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    result = compute_reverse_kl_full_vocab(
        student_logits=student_logits,
        teacher_full_log_probs=teacher_rows,
        teacher_response_indices=response_indices,
        data=data,
        config=config,
    )["distillation_losses"]

    student_log_probs = torch.log_softmax(student_logits[0, [1, 3]], dim=-1)
    expected = (student_log_probs.exp() * (student_log_probs - teacher_rows.values())).sum(dim=-1)
    torch.testing.assert_close(result[0, [1, 3]], expected)
    assert torch.count_nonzero(result[0, [0, 2, 4, 5, 6, 7, 8]]) == 0
    result.sum().backward()
    assert torch.count_nonzero(student_logits.grad[0, [1, 3]]) > 0
    assert torch.count_nonzero(student_logits.grad[0, [0, 2, 4, 5, 6, 7, 8]]) == 0


def test_reverse_kl_full_vocab_settings_are_not_sampled_or_topk():
    config = DistillationLossConfig(
        loss_mode="reverse_kl_full_vocab",
        topk=None,
        use_policy_gradient=False,
        log_prob_min_clamp=None,
        loss_max_clamp=None,
    )

    assert config.loss_settings.use_full_vocab is True
    assert config.loss_settings.use_topk is False
    assert config.loss_settings.use_estimator is False


def test_full_vocab_teacher_config_enables_unlimited_vllm_logprobs():
    teacher = DistillationTeacherModelConfig(inference=RolloutConfig(name="vllm"))

    teacher._validate_teacher_logprobs(use_topk=False, topk=None, use_full_vocab=True)

    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == -1


def test_vllm_full_vocab_prompt_logprobs_are_dense_in_token_id_order():
    row_1 = {
        2: SimpleNamespace(logprob=-3.0),
        0: SimpleNamespace(logprob=-1.0),
        1: SimpleNamespace(logprob=-2.0),
    }
    row_2 = {
        1: SimpleNamespace(logprob=-5.0),
        2: SimpleNamespace(logprob=-6.0),
        0: SimpleNamespace(logprob=-4.0),
    }
    output = SimpleNamespace(prompt_logprobs=[None, row_1, row_2])
    result = {}

    extract_prompt_logprobs(output, num_prompt_logprobs=-1, result_dict=result)

    assert result == {
        "prompt_full_logprobs": [
            [-1.0, -2.0, -3.0],
            [-4.0, -5.0, -6.0],
            [0.0, 0.0, 0.0],
        ]
    }


def test_vllm_full_vocab_sample_logprobs_return_only_generated_positions():
    generated_row = {
        2: SimpleNamespace(logprob=-3.0),
        0: SimpleNamespace(logprob=-1.0),
        1: SimpleNamespace(logprob=-2.0),
    }
    output = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[generated_row])])
    result = {}

    extract_sample_logprobs(output, num_logprobs=-1, result_dict=result)

    assert result == {"sample_full_logprobs": [[-1.0, -2.0, -3.0]]}


class _FakeFullVocabClient:
    def __init__(self):
        self.prompts: list[list[int]] = []
        self.sampling_params: list[dict] = []

    async def generate(self, *, prompt_ids, sampling_params, **kwargs):
        del kwargs
        self.prompts.append(list(prompt_ids))
        self.sampling_params.append(dict(sampling_params))
        row = [-1.0, -2.0, -3.0, -4.0]
        return SimpleNamespace(extra_fields={"sample_full_logprobs": [row]})


def test_full_vocab_teacher_scores_only_selected_next_token_prefixes():
    client = _FakeFullVocabClient()
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = SimpleNamespace(
        loss_settings=SimpleNamespace(use_full_vocab=True)
    )
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(temperature=1.0))
    }
    manager.teacher_client = {"teacher": client}

    response_indices, rows = asyncio.run(
        manager.compute_teacher_full_vocab_logprobs_single(
            sequence_ids=[90, 91, 20, 30, 21],
            teacher_prompt_length=2,
            response_mask=[1, 0, 1],
        )
    )

    # Token 0 is scored from the teacher prompt. Token 2 is scored from the
    # exact original prefix including token 0 and the masked environment token 1.
    assert client.prompts == [[90, 91], [90, 91, 20, 30]]
    assert all(
        params == {"max_tokens": 1, "temperature": 1.0, "logprobs": -1}
        for params in client.sampling_params
    )
    torch.testing.assert_close(response_indices, torch.tensor([0, 2]))
    assert rows.shape == (2, 4)


class _FakeTeacherManager:
    def __init__(self):
        self.critique_prompts: list[list[int]] = []
        self.scored_sequences: list[list[int]] = []
        self.empty_calls: list[int] = []

    async def generate_teacher_critique_single(self, *, prompt_ids, **kwargs):
        del kwargs
        self.critique_prompts.append(list(prompt_ids))
        return [70, 71]

    async def compute_teacher_logprobs_single(self, *, sequence_ids, **kwargs):
        del kwargs
        self.scored_sequences.append(list(sequence_ids))
        length = len(sequence_ids)
        teacher_ids = torch.arange(length, dtype=torch.int32).unsqueeze(-1)
        teacher_logprobs = torch.arange(length, dtype=torch.float32).unsqueeze(-1)
        return teacher_ids, teacher_logprobs

    def empty_teacher_outputs(self, *, sequence_length, pad_token_id):
        self.empty_calls.append(sequence_length)
        return (
            torch.full((sequence_length, 1), pad_token_id, dtype=torch.int32),
            torch.zeros((sequence_length, 1), dtype=torch.float32),
        )


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.decode_calls: list[list[int]] = []
        self.scoring_prompts: list[str] = []

    def __len__(self):
        return 4

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        self.decode_calls.append(list(ids))
        return "<error_step>2</error_step>\n<critique>The second action is the earliest error.</critique>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, tools, return_dict
        assert kwargs.get("enable_thinking") is False
        self.scoring_prompts.append(messages[-1]["content"])
        return [90, 91, 92]


def _make_worker(manager, tokenizer):
    return type(
        "_Worker",
        (),
        {
            "distillation_enabled": True,
            "teacher_key": "data_source",
            "teacher_server_manager": manager,
            "tokenizer": tokenizer,
            "config": SimpleNamespace(data={"apply_chat_template_kwargs": {"enable_thinking": False}}),
        },
    )()


def test_teacher_generates_critique_then_scores_exact_prefix_through_error_step():
    manager = _FakeTeacherManager()
    tokenizer = _FakeTokenizer()
    worker = _make_worker(manager, tokenizer)
    output = AgentLoopOutput(
        prompt_ids=[10, 11, 12, 13],
        teacher_critique_prompt_ids=[80, 81],
        teacher_prompt_template="Task and privileged critique:\n{critique}",
        response_step_end_indices=[2, 6],
        fallback_error_step=0,
        opd_eligible=True,
        response_ids=[20, 21, 30, 31, 22, 23, 40, 41],
        response_mask=[1, 1, 0, 0, 1, 1, 0, 0],
        response_logprobs=[-0.1] * 8,
        metrics=AgentLoopMetrics(),
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            worker,
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    assert manager.critique_prompts == [[80, 81]]
    assert tokenizer.decode_calls == [[70, 71]]
    assert "<error_step>2</error_step>" in tokenizer.scoring_prompts[0]
    # The original response token prefix is passed through directly. Tokens after
    # the teacher-identified second action are not scored or trained.
    assert manager.scored_sequences == [[90, 91, 92, 20, 21, 30, 31, 22, 23]]
    assert output.response_ids == [20, 21, 30, 31, 22, 23]
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert output.response_logprobs == [-0.1] * 6
    assert output.extra_fields["opd_error_step"] == 1
    assert output.extra_fields["opd_response_cutoff"] == 6
    assert output.extra_fields["opd_original_response_length"] == 8
    assert output.extra_fields["opd_selected"] is True

    aligned_ids = output.extra_fields["teacher_ids"]
    aligned_logprobs = output.extra_fields["teacher_logprobs"]
    assert aligned_ids.shape == aligned_logprobs.shape == (10, 1)
    torch.testing.assert_close(aligned_ids[3:9, 0], torch.arange(2, 8, dtype=torch.int32))
    torch.testing.assert_close(aligned_logprobs[3:9, 0], torch.arange(2, 8, dtype=torch.float32))


class _FakeFullVocabTeacherManager(_FakeTeacherManager):
    uses_full_vocab = True

    def __init__(self):
        super().__init__()
        self.full_vocab_calls: list[dict] = []

    async def compute_teacher_full_vocab_logprobs_single(self, **kwargs):
        self.full_vocab_calls.append(kwargs)
        indices = torch.tensor(
            [i for i, value in enumerate(kwargs["response_mask"]) if value], dtype=torch.int64
        )
        return indices, torch.full((indices.numel(), 4), -1.0, dtype=torch.float32)

    def empty_teacher_full_vocab_outputs(self, *, vocab_size):
        return torch.empty((0,), dtype=torch.int64), torch.empty((0, vocab_size), dtype=torch.float32)


def test_critique_conditioned_opd_requests_full_vocab_for_exact_error_prefix_only():
    manager = _FakeFullVocabTeacherManager()
    tokenizer = _FakeTokenizer()
    worker = _make_worker(manager, tokenizer)
    output = AgentLoopOutput(
        prompt_ids=[10, 11, 12, 13],
        teacher_critique_prompt_ids=[80, 81],
        teacher_prompt_template="Task and privileged critique:\n{critique}",
        response_step_end_indices=[2, 6],
        fallback_error_step=0,
        opd_eligible=True,
        response_ids=[20, 21, 30, 31, 22, 23, 40, 41],
        response_mask=[1, 1, 0, 0, 1, 1, 0, 0],
        metrics=AgentLoopMetrics(),
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            worker,
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    assert manager.scored_sequences == []
    assert len(manager.full_vocab_calls) == 1
    call = manager.full_vocab_calls[0]
    assert call["sequence_ids"] == [90, 91, 92, 20, 21, 30, 31, 22, 23]
    assert call["teacher_prompt_length"] == 3
    assert call["response_mask"] == [1, 1, 0, 0, 1, 1]
    torch.testing.assert_close(output.extra_fields["teacher_response_indices"], torch.tensor([0, 1, 4, 5]))
    assert output.extra_fields["teacher_full_logprobs"].shape == (4, 4)
    assert "teacher_ids" not in output.extra_fields
    assert "teacher_logprobs" not in output.extra_fields


def test_successful_trajectory_is_excluded_without_calling_teacher():
    manager = _FakeTeacherManager()
    tokenizer = _FakeTokenizer()
    worker = _make_worker(manager, tokenizer)
    output = AgentLoopOutput(
        prompt_ids=[10, 11],
        teacher_critique_prompt_ids=[],
        opd_eligible=False,
        response_ids=[20, 21, 22],
        response_mask=[1, 0, 1],
        metrics=AgentLoopMetrics(),
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            worker,
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    assert manager.critique_prompts == []
    assert manager.scored_sequences == []
    assert manager.empty_calls == [5]
    assert output.response_mask == [0, 0, 0]
    assert output.extra_fields["opd_selected"] is False
    assert output.extra_fields["teacher_ids"].shape == (5, 1)


def test_malformed_critique_uses_configured_error_step_fallback():
    assert parse_critique_error_step("unstructured critique", num_steps=3, fallback_step=1) == 1


def test_empty_opd_batch_returns_connected_zero_loss():
    student_logprobs = torch.randn(1, 3, requires_grad=True)
    data = TensorDict({"response_mask": torch.zeros((1, 3), dtype=torch.int64)}, batch_size=[1])

    loss, metrics = distillation_ppo_loss(
        config=None,
        distillation_config=None,
        model_output={"log_probs": student_logprobs},
        data=data,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(student_logprobs.grad) == 0
    assert metrics["distillation/empty_opd_batch"] == 1.0

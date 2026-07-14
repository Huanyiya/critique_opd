# Text-only ALFWorld critique-conditioned OPD

This example uses veRL's native agent-loop rollout, teacher inference workers,
prompt-logprob computation, and exact full-vocabulary reverse-KL distillation. The student alone
generates each ALFWorld trajectory. Successful trajectories are retained for
environment-reward metrics but receive a zero training mask. For a failed
trajectory, the same teacher worker performs two operations:

1. Generate privileged critique `c` from the task and the complete failed
   trajectory. The structured critique identifies the earliest erroneous step.
2. Score the original student response token IDs under a prompt containing the
   task and `c`. The response is never decoded and re-tokenized for scoring.

The `reverse_kl_full_vocab` loss compares the complete student and teacher
distributions at the same response-token positions:
`KL(p_student || p_teacher)`. This is not the native sampled-token `kl` estimator
and it does not truncate either distribution to top-k tokens.

The response is truncated at the end of the teacher-identified erroneous action.
Only student-generated action tokens from the start through that step have
`response_mask=1`; later actions are not included in OPD.
If a training batch contains no failed-trajectory tokens, the actor optimizer
step is skipped entirely.

To control memory, full-vocabulary teacher rows are retained only for
student-generated tokens with `response_mask=1`. Teacher-prompt tokens,
environment-observation tokens, successful trajectories, and tokens after the
identified error cutoff do not carry a vocabulary-sized tensor. The teacher
server requests one vLLM `logprobs=-1` next-token row for each selected response
position; the generated dummy token is discarded and never enters the
trajectory. This avoids materializing full-vocabulary rows for the teacher
prompt or environment observations. The teacher engine is configured with
`max_logprobs=-1` automatically. The actor path currently requires FSDP eager
logits, fused kernels disabled, and Ulysses sequence parallel size 1.

## Setup

From the veRL repository root, install the optional environment and download its
official game data:

```bash
uv pip install alfworld gymnasium==0.29.1 stable-baselines3==2.6.0 TransferQueue==0.1.6
alfworld-download -f
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

The loop searches the installed package for `configs/base_config.yaml`. If the
package does not ship that file, point to the official config explicitly:

```bash
export ALFWORLD_CONFIG_PATH=/path/to/ALFWorld/configs/base_config.yaml
```

No OPID-bundled veRL, trainer, advantage estimator, GiGPO code, analyzer, or
skill prompt is required.

## Training

Student and teacher paths are independently configurable. Both should be
Qwen3.5-compatible checkpoints for this launcher.

```bash
STUDENT_MODEL_PATH=/models/Qwen3.5-student \
TEACHER_MODEL_PATH=/models/Qwen3.5-teacher \
bash examples/opd/alfworld/run_qwen35_alfworld_opd.sh
```

The launcher passes `data.apply_chat_template_kwargs.enable_thinking=False`, so
Qwen3.5 thinking is disabled through the tokenizer chat-template interface. The
agent is instructed to emit only `<action>...</action>`.

Pure distillation is the default: task rewards are excluded from the training
loss, while each ALFWorld environment reward remains available in rollout and
evaluation metrics.

By default, the actor/student pool uses four GPUs and the separate teacher pool
uses four GPUs. Override `STUDENT_GPUS_PER_NODE`, `TEACHER_GPUS_PER_NODE`,
`STUDENT_TP`, and `TEACHER_TP` to match the node and model sizes.

## Two-task smoke test

The smoke launcher restricts the driver datasets and ALFWorld game set to two
tasks, runs at most two environment steps, and performs one training epoch:

```bash
STUDENT_MODEL_PATH=/models/Qwen3.5-student \
TEACHER_MODEL_PATH=/models/Qwen3.5-teacher \
bash examples/opd/alfworld/run_qwen35_alfworld_opd_smoke.sh
```

It still requires two GPUs by default: one for the student pool and one for the
teacher pool. Model loading, vLLM teacher scoring, and the FSDP update are
GPU-dependent and are not covered by the CPU unit tests.

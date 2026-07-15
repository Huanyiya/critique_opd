# Text-only ALFWorld critique-conditioned OPD

This example uses veRL's native agent-loop rollout, teacher inference workers,
prompt-logprob computation, and top-k reverse-KL distillation. The student alone
generates each ALFWorld trajectory. Successful trajectories are retained for
environment-reward metrics but receive a zero training mask. For a failed
trajectory, the same teacher worker performs two operations:

1. Generate privileged critique `c` from the task and the complete failed
   trajectory. The structured critique identifies the earliest erroneous step.
2. Score each original student action twice: once under that step's current
   prompt augmented with `c`, and once under the exact original student current
   prompt without `c`. The response is never decoded and re-tokenized.

The `reverse_kl_topk` loss uses the teacher's top 16 token IDs at each response
position. It gathers the student's probabilities at those same IDs and computes
`sum p_student * (log p_student - log p_teacher)` after separately normalizing
the student and teacher distributions on that support. This is a distributional
top-k loss, not the sampled-token `kl` estimator.
The privileged reverse KL drives the student update. The unprivileged reverse
KL is computed in the same forward pass as a control and is logged as
`distillation/unprivileged_reverse_kl`; it is not added to the training loss.

The rollout follows OPID's step-level packing: `data.max_response_length` is the
maximum length of one student action response, not a full-trajectory token
budget. Environment observations and admissible actions are used to build the
next step's current prompt and are recorded in metadata; they are not appended
to `responses`. After teacher critique, the failed trajectory is expanded into
one training sample per student step from the beginning through the
teacher-identified erroneous action. Later actions are not included in OPD.
If a training batch contains no failed-trajectory tokens, the actor optimizer
step is skipped entirely.

The teacher server requests `prompt_logprobs=16`; the generated dummy token is
discarded and never enters the trajectory. Successful trajectories and tokens
after the identified error cutoff remain excluded by `response_mask`. The
launcher uses eager FSDP logits because the fused top-k kernel implements the
native forward-KL path rather than this reverse-KL loss.

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

The data preparation step follows OPID's placeholder-data flow. It writes
`TRAIN_DATA_SIZE` empty-prompt train rows and `VAL_DATA_SIZE` empty-prompt
validation rows; these rows only let veRL build a DataLoader and do not contain
real ALFWorld tasks. Real tasks are sampled inside the ALFWorld agent loop from
the full legal, solvable game pool under `$ALFWORLD_DATA/json_2.1.1`.

By default `TRAIN_DATA_SIZE=16` and `GROUP_SIZE=8`. `data.train_batch_size`
equals `TRAIN_DATA_SIZE`; `actor_rollout_ref.rollout.n` equals `GROUP_SIZE`.
Each global step therefore collects `TRAIN_DATA_SIZE × GROUP_SIZE` trajectories.
The same task group shares one shuffled game iterator and seed, so the 8
trajectories in that group start from the same ALFWorld task while sampling
independent student rollouts. Different task groups use different seeds. Since
the placeholder train set size equals the batch size, one epoch is one training
step; `trainer.total_epochs` is effectively the total number of training steps.

`MAX_RESPONSE_LENGTH` controls the per-step action width. The launcher defaults
`ALFWORLD_MAX_ACTION_TOKENS` to the same value, while `ALFWORLD_MAX_STEPS`
controls the maximum number of environment steps in a trajectory.

## Two-task smoke test

The smoke launcher sets `TRAIN_DATA_SIZE=2`, `GROUP_SIZE=2`, and
`VAL_DATA_SIZE=2`, runs at most two environment steps, and performs one training
epoch:

```bash
STUDENT_MODEL_PATH=/models/Qwen3.5-student \
TEACHER_MODEL_PATH=/models/Qwen3.5-teacher \
bash examples/opd/alfworld/run_qwen35_alfworld_opd_smoke.sh
```

It still requires two GPUs by default: one for the student pool and one for the
teacher pool. Model loading, vLLM teacher scoring, and the FSDP update are
GPU-dependent and are not covered by the CPU unit tests.

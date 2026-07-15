#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

: "${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH to a Qwen3.5 student checkpoint}"
: "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH to a Qwen3.5 teacher checkpoint}"

DATA_DIR=${DATA_DIR:-${HOME}/data/alfworld_opd}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/alfworld_opd}
TRAIN_TASKS=${TRAIN_TASKS:-8}
VAL_TASKS=${VAL_TASKS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

STUDENT_GPUS_PER_NODE=${STUDENT_GPUS_PER_NODE:-4}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
STUDENT_TP=${STUDENT_TP:-1}
TEACHER_TP=${TEACHER_TP:-4}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.8}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
PROJECT_NAME=${PROJECT_NAME:-alfworld_opd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_text_alfworld_opd}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}

export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-30}
export ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH:-5}
export ALFWORLD_MAX_ACTION_TOKENS=${ALFWORLD_MAX_ACTION_TOKENS:-128}
export ALFWORLD_TEACHER_CRITIQUE_MAX_TOKENS=${ALFWORLD_TEACHER_CRITIQUE_MAX_TOKENS:-256}
export ALFWORLD_TEACHER_CRITIQUE_MIN_CONFIDENCE=${ALFWORLD_TEACHER_CRITIQUE_MIN_CONFIDENCE:-0.1}
export ALFWORLD_TEACHER_CRITIQUE_REJECT_LOG_PATH=${ALFWORLD_TEACHER_CRITIQUE_REJECT_LOG_PATH:-"${OUTPUT_DIR}/teacher_critique_rejects.txt"}

python3 "${ROOT_DIR}/examples/opd/alfworld/prepare_data.py" \
    --output-dir "${DATA_DIR}" \
    --train-size "${TRAIN_TASKS}" \
    --val-size "${VAL_TASKS}" \
    --eval-split "${ALFWORLD_EVAL_SPLIT:-eval_in_distribution}"

python3 -m verl.trainer.main_ppo_sync \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${TRAIN_TASKS}" \
    data.val_batch_size="${VAL_TASKS}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_TASKS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${STUDENT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_opd \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${ROOT_DIR}/examples/opd/alfworld/agent_loop.yaml" \
    distillation.enabled=True \
    distillation.n_gpus_per_node="${TEACHER_GPUS_PER_NODE}" \
    distillation.nnodes="${NNODES}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL_PATH}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_MODEL_LEN}" \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    distillation.distillation_loss.loss_mode=reverse_kl_full_vocab \
    distillation.distillation_loss.topk=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${STUDENT_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.logger='[console]' \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.critic_warmup=0 \
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=1 \
    "$@"

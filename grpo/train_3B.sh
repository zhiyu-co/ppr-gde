#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

MODEL_PATH=${MODEL_PATH:-/path/to/base-model}
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"$REPO_ROOT/checkpoints/grpo_nothink_3B"}
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/roleplay"}
export LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log/grpo_nothink_3B"}

trainer_project=${trainer_project:-roleplay_grpo_3B_nothink}
trainer_experiment=${trainer_experiment:-grpo_nothink}

mkdir -p "$LOG_DIR" "$MODEL_SAVE_DIR"

python3 -u -m verl.trainer.main_ppo \
    +has_thinking=False \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val_512.parquet" \
    data.val_char_files="$DATA_DIR/char_rm_val_2048.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_char_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    +actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=0.9 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    "trainer.logger=['wandb']" \
    trainer.project_name="$trainer_project" \
    trainer.experiment_name="$trainer_experiment" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$MODEL_SAVE_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    +trainer.diversity_ratio=0.0 \
    +trainer.val_before_train=False \
    trainer.total_epochs=4 "$@" \
    2>&1 | tee "$LOG_DIR/train.log"

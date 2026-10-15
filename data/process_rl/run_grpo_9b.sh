#!/bin/bash
# GRPO Training with verifier as reward (Qwen3.5-9B)
# Environment: qwen36

export PATH="/path/to/environment/env/qwen36/bin:$PATH"
export OUTPUT_PATH="/path/to/datasets/dataflywheel/output/checkpoints/grpo_9b"
export EXP_NAME="embodiedforge_grpo_9b"

mkdir -p ${OUTPUT_PATH}
set -x
export PYTHONUNBUFFERED=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ADDRESS=""
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Do not remove Ray state owned by unrelated jobs.

# Use SFT checkpoint; fallback to base model
MODEL_PATH="/path/to/datasets/dataflywheel/output/checkpoints/sft_9b"
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    MODEL_PATH="/path/to/environment/hf_models/Qwen/Qwen3.5-9B"
fi

cd /path/to/environment/Rex-Omni/finetuning

python3 -m verl.trainer.main \
    config=verl/configs/config.yaml \
    data.config_path=/path/to/datasets/dataflywheel/process_rl/configs/grpo_9b.py \
    data.format_prompt=verl/configs/r1v_format.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=4 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=50 \

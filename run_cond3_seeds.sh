#!/bin/bash
# 条件3(3枚並べ336)のシード違い追加実行(2026-09-18): seed=1, seed=2
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/run/media/tatsu/RAID10_Data/hf_cache
export HF_HUB_DISABLE_XET=1

source /home/tatsu/miniconda3/bin/activate clip_eval
cd /run/media/tatsu/RAID10_Data/ResearchAI/research-2/code

RESULTS_ROOT=/run/media/tatsu/RAID10_Data/ResearchAI/research-2/runs_format_ablation_20260916

run_one() {
  local name=$1
  local seed=$2

  echo "=========================================="
  echo "[$(date -Iseconds)] START $name (format=three_panel clip_model=clip-vit-large-patch14-336 seed=$seed)"
  echo "=========================================="

  python3 main_opt_format_ablation.py \
    --seed "$seed" \
    --save_dir "$RESULTS_ROOT/${name}" \
    --substrate h_lenia \
    --rollout_steps 500 \
    --format three_panel \
    --foundation_model clip \
    --clip_model clip-vit-large-patch14-336 \
    --img_size 336 \
    --pop_size 30 \
    --n_iters 100 \
    --sigma 0.1 \
    2>&1 | tee "$RESULTS_ROOT/${name}.log"

  echo "[$(date -Iseconds)] DONE $name (exit=${PIPESTATUS[0]})"
}

run_one "3_three_panel_336_seed1" 1
run_one "3_three_panel_336_seed2" 2

echo "=== Condition3 seed variation (seed=1,2) complete ==="

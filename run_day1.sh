#!/bin/bash
# 本番実行 1日目(2026-09-16): 条件2 -> 3 -> 1 の順、GPU1限定
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
  local format=$2
  local clip_model=$3
  local img_size=$4

  echo "=========================================="
  echo "[$(date -Iseconds)] START $name (format=$format clip_model=$clip_model img_size=$img_size)"
  echo "=========================================="

  python3 main_opt_format_ablation.py \
    --seed 0 \
    --save_dir "$RESULTS_ROOT/${name}" \
    --substrate h_lenia \
    --rollout_steps 500 \
    --format "$format" \
    --foundation_model clip \
    --clip_model "$clip_model" \
    --img_size "$img_size" \
    --pop_size 30 \
    --n_iters 100 \
    --sigma 0.1 \
    2>&1 | tee "$RESULTS_ROOT/${name}.log"

  echo "[$(date -Iseconds)] DONE $name (exit=${PIPESTATUS[0]})"
}

run_one "2_rgb_overlay_336" rgb_overlay clip-vit-large-patch14-336 336
run_one "3_three_panel_336" three_panel clip-vit-large-patch14-336 336
run_one "1_grayscale_336"   grayscale   clip-vit-large-patch14-336 336

echo "=== Day1 (conditions 2,3,1) complete ==="

#!/bin/bash
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/run/media/tatsu/RAID10_Data/hf_cache
export HF_HUB_DISABLE_XET=1
source /home/tatsu/miniconda3/bin/activate clip_eval
cd /run/media/tatsu/RAID10_Data/ResearchAI/research-2/code
RESULTS_ROOT=/run/media/tatsu/RAID10_Data/ResearchAI/research-2/runs_format_ablation_20260916

echo "[$(date -Iseconds)] START 5_grayscale_224ctrl"
python3 main_opt_format_ablation.py \
  --seed 0 \
  --save_dir "$RESULTS_ROOT/5_grayscale_224ctrl" \
  --substrate h_lenia --rollout_steps 500 \
  --format grayscale --foundation_model clip \
  --clip_model clip-vit-large-patch14 --img_size 336 \
  --pop_size 30 --n_iters 100 --sigma 0.1 \
  2>&1 | tee "$RESULTS_ROOT/5_grayscale_224ctrl.log"
echo "[$(date -Iseconds)] DONE 5_grayscale_224ctrl (exit=${PIPESTATUS[0]})"

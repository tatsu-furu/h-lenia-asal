#!/bin/bash
# 較正実行(2026-09-16): 5条件 x 2世代のみ、GPU1(3090)限定で実測時間・VRAMを測定
#
# 注意: set -e は使わない。1条件が失敗しても残りの条件を続行し、
# FAILED として記録する(失敗検知のために全体を止めない)。
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/run/media/tatsu/RAID10_Data/hf_cache
export HF_HUB_DISABLE_XET=1

source /home/tatsu/miniconda3/bin/activate clip_eval
cd /run/media/tatsu/RAID10_Data/ResearchAI/research-2/code

CALIB_ROOT=/run/media/tatsu/RAID10_Data/ResearchAI/research-2/calibration_20260916
mkdir -p "$CALIB_ROOT"

run_one() {
  local name=$1
  local format=$2
  local clip_model=$3
  local img_size=$4

  echo "=========================================="
  echo "Condition: $name (format=$format clip_model=$clip_model img_size=$img_size)"
  echo "=========================================="

  local vram_log="$CALIB_ROOT/${name}_vram.log"
  : > "$vram_log"

  # nvidia-smi単体の--loop機能を使う(サブシェル+whileより単一プロセスでkillが確実)
  nvidia-smi -l 2 --query-gpu=memory.used --format=csv,noheader,nounits -i 1 > "$vram_log" 2>/dev/null &
  local monitor_pid=$!

  local t0=$(date +%s)
  timeout 1800 python3 main_opt_format_ablation.py \
    --seed 0 \
    --save_dir "$CALIB_ROOT/${name}" \
    --substrate h_lenia \
    --rollout_steps 500 \
    --format "$format" \
    --foundation_model clip \
    --clip_model "$clip_model" \
    --img_size "$img_size" \
    --pop_size 30 \
    --n_iters 2 \
    --sigma 0.1 \
    > "$CALIB_ROOT/${name}.log" 2>&1
  local exit_code=$?
  local t1=$(date +%s)

  kill "$monitor_pid" 2>/dev/null
  wait "$monitor_pid" 2>/dev/null
  true  # wait/killの終了コードで後続を止めない

  local elapsed=$((t1 - t0))
  local peak_vram
  peak_vram=$(sort -n "$vram_log" 2>/dev/null | tail -1)
  local status="OK"
  if [ "$exit_code" -ne 0 ]; then
    status="FAILED(exit=$exit_code)"
  fi
  echo "$name status=$status elapsed_sec=$elapsed peak_vram_MiB=$peak_vram" | tee -a "$CALIB_ROOT/summary.txt"
}

echo "condition status elapsed_sec peak_vram_MiB" > "$CALIB_ROOT/summary.txt"

run_one "1_grayscale_336"   grayscale   clip-vit-large-patch14-336 336
run_one "2_rgb_overlay_336" rgb_overlay clip-vit-large-patch14-336 336
run_one "3_three_panel_336" three_panel clip-vit-large-patch14-336 336
run_one "4_four_panel_336"  four_panel  clip-vit-large-patch14-336 336
run_one "5_grayscale_224ctrl" grayscale clip-vit-large-patch14 336

echo "=== calibration done ==="
cat "$CALIB_ROOT/summary.txt"

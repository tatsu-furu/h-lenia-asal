# H-Lenia ASAL

Automated Search for Artificial Life (ASAL) applied to H-Lenia.  
Uses CLIP (ViT-Large/14) as fitness function with CMA-ES to discover hierarchical microorganism patterns.

## Overview

Applies the ASAL framework (Kumar et al., 2024) to H-Lenia's 3-layer system.  
Each candidate's simulation state is rendered as an RGB Overlay image (L1=R, L2=G, L3=B) and evaluated by CLIP cosine similarity against a text prompt.

**Base model**: [tatsu-furu/h-lenia](https://github.com/tatsu-furu/h-lenia)

## Key Result

After 100 generations of CMA-ES search (λ=30, seed=0):
- Best fitness: **L = −0.3280** (CLIP similarity ≈ 0.328)
- Discovered a self-swimming hierarchical microorganism with:
  - L1: filament-like fine structure (organelle level)
  - L2: mid-scale cluster skeleton (cytoskeleton level)  
  - L3: large-scale organized shape (whole-body level)
- Transfer Entropy: bottom-up dominant (L1→L2: 0.42, L2→L3: 0.38 vs L3→L2: 0.15, L2→L1: 0.11)

## File Structure

| File | Role |
|------|------|
| `main_opt_multilayer_overlay_simple.py` | **Main**: CMA-ES optimization loop |
| `create_multi_generation_animations.py` | Generate `gen_XXXX_2x2_layers.mp4` animations |
| `rollout.py` | Simulation rollout |
| `asal_metrics.py` | Fitness / metrics computation |
| `util.py` | Utilities (save/load pkl, json) |
| `substrates/h_lenia*.py` | H-Lenia simulation engine |
| `foundation_models/clip.py` | CLIP fitness evaluation |

## Quick Start

```bash
pip install -r requirements.txt

python main_opt_multilayer_overlay_simple.py \
    --seed 0 \
    --save_dir my_experiment \
    --substrate h_lenia \
    --rollout_steps 500 \
    --foundation_model clip \
    --clip_model clip-vit-large-patch14 \
    --img_size 336 \
    --prompts "Artificial microorganisms with a hierarchical structure consisting of multiple scales, as seen under a microscope" \
    --pop_size 30 \
    --n_iters 100 \
    --sigma 0.1
```

## Reproduce Paper Results

```bash
# Optimization (seed=0, ~3h14m on GPU)
python main_opt_multilayer_overlay_simple.py \
    --seed 0 --save_dir experiment_overlay_hierarchical_microscope \
    --img_size 336 --pop_size 30 --n_iters 100 --sigma 0.1

# Generate animations
python create_multi_generation_animations.py \
    --exp_dir experiment_overlay_hierarchical_microscope \
    --range 0:15
```

## Experiment Settings (Paper)

| Parameter | Value |
|-----------|-------|
| H-Lenia layers | L1: 1536×1536, L2: 384×384, L3: 96×96 |
| Time scales | T_S1=1, T_S2=4, T_S3=16 |
| Inter-layer interaction | k21=0.6, k12=0.5, k32=0.9, k23=0.4 (fixed) |
| Optimized params | growth function (μ, σ) offset × 3 layers = 6-dim |
| CLIP model | ViT-Large/14, 336px |
| CMA-ES | λ=30, 100 generations, σ₀=0.1 |
| Seed | 0 |

## Requirements
- Python 3.10+
- JAX (GPU recommended)
- evosax (CMA-ES)
- transformers / CLIP
- h5py, numpy, matplotlib, Pillow, tqdm

## References
- Kumar et al. (2024): ASAL
- Hansen (2016): CMA-ES
- Radford et al. (2021): CLIP
- Furukawa (2025): H-Lenia

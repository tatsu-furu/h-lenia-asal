#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 3×3 channel grid and 2×2 layer grid animations for H-Lenia evolution.
Supports multiple generation selection modes:
- Range mode: Generate animations for generations n to m
- List mode: Generate animations for specific generations
- Best mode: Generate animation for the best fitness generation

Version: 3.0 (Multi-generation support)
"""

import sys
sys.path.insert(0, '.')

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import gc
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from PIL import Image
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter

from substrates.h_lenia import HLenia


def create_black_to_color_cmap(color: str):
    """黒から指定された純色への、鮮やかなカスタムカラーマップを作成する"""
    if color.lower() == 'red':
        cdict = {'red':   [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                 'green': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                 'blue':  [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    elif color.lower() == 'green':
        cdict = {'red':   [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                 'green': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                 'blue':  [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    elif color.lower() == 'blue':
        cdict = {'red':   [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                 'green': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                 'blue':  [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}
    else:
        raise ValueError("Color must be 'red', 'green', or 'blue'")
    
    return mcolors.LinearSegmentedColormap('custom_cmap', cdict)


def load_params_from_generation(gen_dir):
    """Load parameters from generation directory"""
    params_path = os.path.join(gen_dir, 'params.pkl')
    
    if not os.path.exists(params_path):
        return None, None, None
    
    with open(params_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(data, dict):
        best_params = data.get('best_params', data)
        iteration = data.get('iteration', 0)
        fitness = data.get('best_fitness', 0.0)
    elif isinstance(data, tuple):
        best_params = data[0]
        iteration = 0
        fitness = data[1] if len(data) > 1 else 0.0
    else:
        best_params = data
        iteration = 0
        fitness = 0.0
    
    return best_params, iteration, fitness


def find_best_generation(base_dir):
    """
    Find the generation with the best fitness from data_save.pkl
    Returns: (best_gen_idx, best_fitness) or (None, None) if not found
    """
    data_save_path = os.path.join(base_dir, 'data_save.pkl')
    
    if not os.path.exists(data_save_path):
        print(f"  ⚠️  data_save.pkl not found in {base_dir}")
        return None, None
    
    try:
        with open(data_save_path, 'rb') as f:
            data = pickle.load(f)
        
        best_losses = np.array(data['best_loss'])
        generations = np.array(data['generation'])
        
        best_idx = np.argmin(best_losses)
        best_gen = int(generations[best_idx])
        best_fitness = float(best_losses[best_idx])
        
        print(f"\n{'='*80}")
        print(f"Best Generation Analysis")
        print(f"{'='*80}")
        print(f"  Best generation: {best_gen}")
        print(f"  Best fitness: {best_fitness:.6f}")
        print(f"  Total generations: {len(generations)}")
        print(f"{'='*80}\n")
        
        return best_gen, best_fitness
        
    except Exception as e:
        print(f"  ⚠️  Error loading data_save.pkl: {e}")
        return None, None


def run_simulation_chunked(params, rollout_steps=8000, chunk_size=50, seed=42):
    """
    Run H-Lenia simulation in memory-efficient chunks.
    Returns:
    - layer_frames: dict with 'A1', 'A2', 'A3' keys containing (T, H, W, C) arrays
    """
    substrate = HLenia()
    rng = random.PRNGKey(seed)
    
    # Initialize state
    rng, init_rng = random.split(rng)
    initial_state = substrate.init_state(init_rng, params)
    
    # Prepare to collect frames for each layer
    A1_frames = []
    A2_frames = []
    A3_frames = []
    
    # Chunk loop
    sim_state = initial_state
    num_chunks = (rollout_steps + chunk_size - 1) // chunk_size
    
    print(f"    Running {rollout_steps} steps in {num_chunks} chunks of {chunk_size}...")
    
    for chunk_idx in tqdm(range(num_chunks), desc="    Simulating", unit="chunk", leave=False):
        start_step = chunk_idx * chunk_size
        steps_in_chunk = min(chunk_size, rollout_steps - start_step)
        
        if steps_in_chunk <= 0:
            continue
        
        # Define scan body for this chunk
        def scan_body(carry_state, chunk_rng):
            next_state = substrate.step_state(chunk_rng, carry_state, params)
            return next_state, next_state
        
        # Generate RNG keys for this chunk
        rng, *chunk_rngs = random.split(rng, steps_in_chunk + 1)
        chunk_rngs = jnp.array(chunk_rngs)
        
        # Run scan for this chunk
        final_state, states = jax.lax.scan(scan_body, sim_state, chunk_rngs)
        
        # Extract frames when L3 was updated (every 16 steps)
        l3_time_scale = 16
        for i in range(steps_in_chunk):
            frame_num = start_step + i
            
            # Check if L3 was updated at this step
            if frame_num % l3_time_scale != 0:
                continue
            
            state = jax.tree.map(lambda x: x[i], states)
            
            # Move each layer to CPU immediately (keep full resolution and channels)
            A1_np = np.asarray(jax.device_get(state["A1"]))
            A2_np = np.asarray(jax.device_get(state["A2"]))
            A3_np = np.asarray(jax.device_get(state["A3"]))
            
            A1_frames.append(A1_np)
            A2_frames.append(A2_np)
            A3_frames.append(A3_np)
        
        # Update state for next chunk
        sim_state = final_state
        
        # Clear GPU memory
        del states, final_state
        jax.clear_caches()
    
    # Stack frames
    layer_frames = {
        'A1': np.stack(A1_frames, axis=0),  # (T, H, W, C)
        'A2': np.stack(A2_frames, axis=0),
        'A3': np.stack(A3_frames, axis=0)
    }
    
    print(f"    Generated {len(A1_frames)} frames")
    
    return layer_frames


def create_3x3_channel_animation(layer_frames, output_path, title="Generation", fps=10, dpi=150):
    """
    Create 3×3 grid animation showing all layers and channels separately
    """
    print(f"    Creating 3×3 channel grid animation...")
    
    A1_data = layer_frames['A1']
    A2_data = layer_frames['A2']
    A3_data = layer_frames['A3']
    
    num_frames = A1_data.shape[0]
    
    # Create 3×3 grid figure
    fig, axes = plt.subplots(3, 3, figsize=(24, 24), dpi=dpi)
    
    layer_names = ['L1 (Fast/1536×1536)', 'L2 (Med/384×384)', 'L3 (Slow/96×96)']
    channel_names = ['Channel 0 (R)', 'Channel 1 (G)', 'Channel 2 (B)']
    cmaps = ['Reds', 'Greens', 'Blues']
    layers_data = [A1_data, A2_data, A3_data]
    
    # Initialize images
    imgs = {}
    for row, (layer_name, layer_data) in enumerate(zip(layer_names, layers_data)):
        for col, (ch_name, cmap) in enumerate(zip(channel_names, cmaps)):
            ax = axes[row, col]
            ch_data = layer_data[0, :, :, col]
            
            im = ax.imshow(ch_data, cmap=cmap, vmin=0, vmax=1, interpolation='none')
            
            title_str = f'{layer_name}\n{ch_name}'
            if row == 0:
                ax.set_title(title_str, fontsize=16, fontweight='bold', pad=8)
            else:
                ax.set_title(title_str, fontsize=14, fontweight='bold', pad=8)
            
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            
            imgs[(row, col)] = im
    
    # Title
    title_text = fig.suptitle(
        f'{title} - 3×3 Channel Grid (Frame 0)\n'
        f'Rows: Layers (L1=Fast, L2=Medium, L3=Slow) | Columns: Channels (R, G, B)',
        fontsize=20, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Animation update
    progress = tqdm(total=num_frames, desc="    Rendering 3×3", unit="frame", leave=False)
    
    def animate(frame_idx):
        for row, layer_data in enumerate(layers_data):
            for col in range(3):
                ch_data = layer_data[frame_idx, :, :, col]
                imgs[(row, col)].set_data(ch_data)
        
        title_text.set_text(
            f'{title} - 3×3 Channel Grid (Frame {frame_idx})\n'
            f'Rows: Layers (L1=Fast, L2=Medium, L3=Slow) | Columns: Channels (R, G, B)'
        )
        
        progress.update(1)
        return list(imgs.values()) + [title_text]
    
    # Create and save animation
    anim = FuncAnimation(fig, animate, frames=num_frames, 
                        interval=1000/fps, blit=True, repeat=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    progress.close()
    plt.close(fig)
    
    filesize_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      ✅ Saved: {output_path} ({filesize_mb:.1f} MB)")
    
    return output_path


def create_2x2_layer_animation(layer_frames, output_path, title="Generation", fps=10):
    """
    Create 2×2 grid animation showing layers + overlay
    """
    print(f"    Creating 2×2 layer grid animation...")
    
    A1_data = layer_frames['A1']
    A2_data = layer_frames['A2']
    A3_data = layer_frames['A3']
    
    num_frames = A1_data.shape[0]
    max_size = max(A1_data[0].shape[0], A2_data[0].shape[0], A3_data[0].shape[0])
    
    # Create custom colormaps
    cmap_red = create_black_to_color_cmap('red')
    cmap_green = create_black_to_color_cmap('green')
    cmap_blue = create_black_to_color_cmap('blue')
    
    # Helper function to render a single frame
    def render_frame(frame_idx):
        from PIL import Image as PILImage
        
        # Convert each layer to grayscale (average across channels)
        a1_gray = np.mean(A1_data[frame_idx], axis=-1)
        a2_gray = np.mean(A2_data[frame_idx], axis=-1)
        a3_gray = np.mean(A3_data[frame_idx], axis=-1)
        
        # Clip to [0, 1]
        a1_gray = np.clip(a1_gray, 0, 1)
        a2_gray = np.clip(a2_gray, 0, 1)
        a3_gray = np.clip(a3_gray, 0, 1)
        
        # Resize all to same size (grayscale)
        a1_img = PILImage.fromarray((a1_gray * 255).astype(np.uint8))
        a2_img = PILImage.fromarray((a2_gray * 255).astype(np.uint8))
        a3_img = PILImage.fromarray((a3_gray * 255).astype(np.uint8))
        
        a1_resized = np.array(a1_img.resize((max_size, max_size), PILImage.BILINEAR)) / 255.0
        a2_resized = np.array(a2_img.resize((max_size, max_size), PILImage.BILINEAR)) / 255.0
        a3_resized = np.array(a3_img.resize((max_size, max_size), PILImage.BILINEAR)) / 255.0
        
        # Create RGB overlay: L1=Red, L2=Green, L3=Blue
        overlay_rgb = np.stack([a1_resized, a2_resized, a3_resized], axis=-1)
        overlay_rgb = np.clip(overlay_rgb, 0, 1)
        
        return {
            'L1': a1_resized,
            'L2': a2_resized,
            'L3': a3_resized,
            'overlay': overlay_rgb
        }
    
    # Create 2×2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
    
    # Initialize with first frame
    first_frame = render_frame(0)
    
    # L1: Red colormap (black to red)
    im_l1 = axes[0, 0].imshow(first_frame['L1'], cmap=cmap_red, vmin=0, vmax=1)
    axes[0, 0].set_title('L1 (Fast Layer - Red)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    cbar_l1 = plt.colorbar(im_l1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar_l1.ax.tick_params(labelsize=10)
    
    # L2: Green colormap (black to green)
    im_l2 = axes[0, 1].imshow(first_frame['L2'], cmap=cmap_green, vmin=0, vmax=1)
    axes[0, 1].set_title('L2 (Medium Layer - Green)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar_l2 = plt.colorbar(im_l2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar_l2.ax.tick_params(labelsize=10)
    
    # L3: Blue colormap (black to blue)
    im_l3 = axes[1, 0].imshow(first_frame['L3'], cmap=cmap_blue, vmin=0, vmax=1)
    axes[1, 0].set_title('L3 (Slow Layer - Blue)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar_l3 = plt.colorbar(im_l3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_l3.ax.tick_params(labelsize=10)
    
    # Overlay: RGB (L1=R, L2=G, L3=B)
    im_overlay = axes[1, 1].imshow(first_frame['overlay'])
    axes[1, 1].set_title('Overlay (L1:R, L2:G, L3:B)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    title_text = fig.suptitle(f'{title} - 2×2 Layer Grid (Frame 0)', 
                              fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    del first_frame
    
    # Animation update
    progress = tqdm(total=num_frames, desc="    Rendering 2×2", unit="frame", leave=False)
    
    def animate(frame_idx):
        # Render frame on-the-fly
        frame_data = render_frame(frame_idx)
        
        im_l1.set_array(frame_data['L1'])
        im_l2.set_array(frame_data['L2'])
        im_l3.set_array(frame_data['L3'])
        im_overlay.set_array(frame_data['overlay'])
        
        title_text.set_text(f'{title} - 2×2 Layer Grid (Frame {frame_idx})')
        
        progress.update(1)
        return [im_l1, im_l2, im_l3, im_overlay, title_text]
    
    # Create and save animation
    anim = FuncAnimation(fig, animate, frames=num_frames, 
                        interval=1000/fps, blit=True, repeat=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer)
    
    progress.close()
    plt.close(fig)
    
    gc.collect()
    
    filesize_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      ✅ Saved: {output_path} ({filesize_mb:.1f} MB)")
    
    return output_path


def process_generation(gen_dir, gen_idx, output_dir, rollout_steps=8000, fps=10, 
                      create_3x3=True, create_2x2=True):
    """Process a single generation: load params, simulate, create animations"""
    
    print(f"\n{'='*80}")
    print(f"Processing Generation {gen_idx}")
    print(f"{'='*80}")
    
    # Load parameters
    print("  Loading parameters...")
    params, iteration, fitness = load_params_from_generation(gen_dir)
    
    if params is None:
        print(f"  ❌ params.pkl not found in {gen_dir}")
        return False
    
    print(f"    Iteration: {iteration}")
    print(f"    Fitness: {fitness:.6f}")
    
    # Run simulation
    print(f"  Running simulation ({rollout_steps} steps)...")
    try:
        layer_frames = run_simulation_chunked(
            params=params,
            rollout_steps=rollout_steps,
            chunk_size=50,
            seed=gen_idx
        )
    except Exception as e:
        print(f"  ❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create animations
    title = f'Generation {gen_idx} (Fitness: {fitness:.4f})'
    success = True
    
    # 1. Create 3×3 channel animation
    if create_3x3:
        output_3x3 = os.path.join(output_dir, f'gen_{gen_idx:04d}_3x3_channels.mp4')
        try:
            create_3x3_channel_animation(layer_frames, output_3x3, title=title, fps=fps, dpi=150)
        except Exception as e:
            print(f"  ❌ 3×3 animation failed: {e}")
            success = False
    
    # 2. Create 2×2 layer animation
    if create_2x2:
        output_2x2 = os.path.join(output_dir, f'gen_{gen_idx:04d}_2x2_layers.mp4')
        try:
            create_2x2_layer_animation(layer_frames, output_2x2, title=title, fps=fps)
        except Exception as e:
            print(f"  ❌ 2×2 animation failed: {e}")
            success = False
    
    # Clean up
    del layer_frames
    gc.collect()
    jax.clear_caches()
    
    if success:
        print(f"  ✅ Generation {gen_idx} complete!")
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Create H-Lenia evolution animations')
    
    # Experiment directory
    parser.add_argument('--exp_dir', type=str, 
                       default='experiment_multilayer_enhanced_highres_reproduction',
                       help='Experiment directory name')
    
    # Generation selection modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--range', type=str, metavar='START:END',
                           help='Generate animations for generations START to END (e.g., 0:15)')
    mode_group.add_argument('--list', type=str, metavar='G1,G2,G3',
                           help='Generate animations for specific generations (e.g., 0,25,50,75,99)')
    mode_group.add_argument('--best', action='store_true',
                           help='Generate animation for the best fitness generation')
    
    # Animation options
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: exp_dir/animations)')
    parser.add_argument('--rollout_steps', type=int, default=8000,
                       help='Simulation steps (default: 8000)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation FPS (default: 10)')
    parser.add_argument('--skip_3x3', action='store_true',
                       help='Skip 3×3 channel animation creation')
    parser.add_argument('--skip_2x2', action='store_true',
                       help='Skip 2×2 layer animation creation')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Support both relative and absolute paths
    if os.path.isabs(args.exp_dir):
        base_dir = args.exp_dir
    else:
        base_dir = os.path.join(script_dir, args.exp_dir)
    
    generations_dir = os.path.join(base_dir, 'generations')
    
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, 'animations')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine generation list
    gen_list = []
    
    if args.range:
        # Range mode: n:m
        try:
            start, end = map(int, args.range.split(':'))
            gen_list = list(range(start, end + 1))
            mode_desc = f"Range {start} to {end}"
        except ValueError:
            print(f"❌ Invalid range format: {args.range}. Use START:END (e.g., 0:15)")
            return
    
    elif args.list:
        # List mode: comma-separated
        try:
            gen_list = [int(g.strip()) for g in args.list.split(',')]
            mode_desc = f"Specific generations: {gen_list}"
        except ValueError:
            print(f"❌ Invalid list format: {args.list}. Use G1,G2,G3 (e.g., 0,25,50,99)")
            return
    
    elif args.best:
        # Best mode: find from data_save.pkl
        best_gen, best_fitness = find_best_generation(base_dir)
        if best_gen is None:
            print("❌ Could not determine best generation")
            return
        gen_list = [best_gen]
        mode_desc = f"Best generation ({best_gen}, fitness={best_fitness:.6f})"
    
    # Print configuration
    print("="*80)
    print("H-Lenia Evolution Animation Generator")
    print("="*80)
    print(f"Experiment: {args.exp_dir}")
    print(f"Mode: {mode_desc}")
    print(f"Generations to process: {len(gen_list)}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"FPS: {args.fps}")
    print(f"Create 3×3: {not args.skip_3x3}")
    print(f"Create 2×2: {not args.skip_2x2}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Process generations
    success_count = 0
    failed_gens = []
    
    for gen_idx in tqdm(gen_list, desc="Overall Progress", unit="gen"):
        gen_dir = os.path.join(generations_dir, f'gen_{gen_idx:04d}')
        
        if not os.path.exists(gen_dir):
            print(f"\n⚠️  Generation {gen_idx} directory not found: {gen_dir}")
            failed_gens.append(gen_idx)
            continue
        
        success = process_generation(
            gen_dir=gen_dir,
            gen_idx=gen_idx,
            output_dir=output_dir,
            rollout_steps=args.rollout_steps,
            fps=args.fps,
            create_3x3=not args.skip_3x3,
            create_2x2=not args.skip_2x2
        )
        
        if success:
            success_count += 1
        else:
            failed_gens.append(gen_idx)
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total requested: {len(gen_list)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_gens)}")
    
    if failed_gens:
        print(f"Failed generations: {failed_gens}")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

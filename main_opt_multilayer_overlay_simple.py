"""
ASAL Optimization: Overlay Image Only (Simplified)

Simple version using only overlay image (average of L1+L2+L3) for CLIP evaluation.
Based on original main_opt.py approach with memory-efficient sequential processing.

Date: 2026-02-14
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial
import gc
from PIL import Image

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
import asal_metrics
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='h_lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=500, help="number of rollout timesteps")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--clip_model", type=str, default="clip-vit-large-patch14", 
                  help="CLIP model variant (clip-vit-large-patch14 for 336px)")
group.add_argument("--img_size", type=int, default=336, help="image resolution (224 or 336)")
group.add_argument("--prompts", type=str, 
                  default="Artificial microorganisms with a hierarchical structure consisting of multiple scales, as seen under a microscope",
                  help="prompts separated by ';'")

group = parser.add_argument_group("optimization")
group.add_argument("--pop_size", type=int, default=30, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=100, help="number of iterations")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


def render_state_overlay(state, params, substrate, img_size=224):
    """
    Render state into single overlay grayscale image (average of L1+L2+L3).
    
    Returns: (H, W, 3) RGB image
    """
    # Extract densities (average over channels)
    a1_density = jnp.mean(state["A1"], axis=-1)  # (H1, W1)
    a2_density = jnp.mean(state["A2"], axis=-1)  # (H2, W2)
    a3_density = jnp.mean(state["A3"], axis=-1)  # (H3, W3)
    
    # Resize to target size
    a1_resized = jax.image.resize(a1_density, (img_size, img_size), method="linear")
    a2_resized = jax.image.resize(a2_density, (img_size, img_size), method="linear")
    a3_resized = jax.image.resize(a3_density, (img_size, img_size), method="linear")
    
    # Clip to [0, 1]
    a1_resized = jnp.clip(a1_resized, 0.0, 1.0)
    a2_resized = jnp.clip(a2_resized, 0.0, 1.0)
    a3_resized = jnp.clip(a3_resized, 0.0, 1.0)
    
    # Overlay: average of all layers
    overlay = (a1_resized + a2_resized + a3_resized) / 3.0
    
    # Create grayscale image (replicate to 3 channels for CLIP)
    img_overlay = jnp.stack([overlay, overlay, overlay], axis=-1)
    
    return img_overlay


def main(args):
    # Validate arguments
    assert args.img_size in [224, 336], "img_size must be 224 or 336"
    
    prompts = args.prompts.split(";")
    print(f"Configuration:")
    print(f"  Substrate: {args.substrate}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Image size: {args.img_size}")
    print(f"  Evaluation: OVERLAY IMAGE ONLY (L1+L2+L3 average)")
    print(f"  Prompts:")
    for p in prompts:
        print(f"    - {p}")
    print(f"  Population size: {args.pop_size}")
    print(f"  Iterations: {args.n_iters}")
    print(args)
    
    # Initialize models
    from foundation_models.clip import CLIP
    fm = CLIP(clip_model=args.clip_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    # Setup save directory
    if args.save_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f'experiment_multilayer_overlay_simple_{timestamp}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    generations_dir = os.path.join(args.save_dir, 'generations')
    os.makedirs(generations_dir, exist_ok=True)
    
    # Save args as JSON
    args_dict = vars(args)
    util.save_json(args.save_dir, "args", args_dict)
    print(f"Saving to: {args.save_dir}")
    
    # Encode prompts
    z_txt = fm.embed_txt(prompts)
    print(f"Text embeddings shape: {z_txt.shape}")
    
    # Create rollout function
    def rollout_final_overlay(rng, params):
        """Run simulation and return overlay image"""
        # Initialize state
        rng_init, rng_sim = split(rng)
        s0 = substrate.init_state(rng_init, params)
        
        # Simulate
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        
        state_final, _ = jax.lax.scan(step_fn, s0, split(rng_sim, args.rollout_steps))
        
        # Render overlay image
        img_overlay = render_state_overlay(state_final, params, substrate, img_size=args.img_size)
        
        return img_overlay, state_final
    
    rollout_final_overlay_jit = jax.jit(rollout_final_overlay)
    
    def calc_loss_single(rng, params):
        """Calculate loss using overlay image"""
        # Run simulation
        img_overlay, state_final = rollout_final_overlay_jit(rng, params)
        
        # Get embedding for overlay image
        z = fm.embed_img(img_overlay)
        
        # Prompt loss
        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        
        loss = loss_prompt
        
        loss_dict = dict(
            loss=loss,
            loss_prompt=loss_prompt
        )
        
        # Cleanup
        del img_overlay, state_final, z
        gc.collect()
        
        return loss, loss_dict
    
    # JIT compile the loss function
    calc_loss_jit = jax.jit(calc_loss_single)
    
    def calc_loss_population(rng, params_pop):
        """Evaluate population sequentially"""
        losses = []
        loss_dicts = []
        
        for i in range(params_pop.shape[0]):
            params_i = jax.tree_util.tree_map(lambda x: x[i], params_pop)
            rng, _rng = split(rng)
            loss, loss_dict = calc_loss_jit(_rng, params_i)
            losses.append(loss)
            loss_dicts.append(loss_dict)
            
            # Periodic cleanup
            if (i + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()
        
        losses = jnp.array(losses)
        
        aggregated_dict = {
            'loss': losses,
            'loss_prompt': jnp.array([d['loss_prompt'] for d in loss_dicts])
        }
        
        return losses, aggregated_dict
    
    # Initialize Sep-CMA-ES
    strategy = evosax.Sep_CMA_ES(
        popsize=args.pop_size,
        num_dims=substrate.num_params,
        elite_ratio=0.5
    )
    
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = split(rng)
    
    mean_init = (substrate.param_max + substrate.param_min) / 2.0
    es_params = strategy.default_params.replace(sigma_init=args.sigma)
    es_state = strategy.initialize(_rng, es_params)
    es_state = es_state.replace(mean=mean_init)
    
    # Storage
    data_save = {
        'best_loss': [],
        'mean_loss': [],
        'best_loss_prompt': [],
        'best_params': [],
        'generation': []
    }
    
    # Evolution loop
    print("\n" + "="*80)
    print("Starting Evolution (Overlay Image Only)")
    print("="*80)
    
    for i_iter in tqdm(range(args.n_iters), desc="Evolution"):
        rng, _rng = split(rng)
        
        # Ask
        params_ask, es_state = strategy.ask(_rng, es_state, es_params)
        params_ask = jnp.clip(params_ask, substrate.param_min, substrate.param_max)
        
        # Evaluate
        rng, _rng = split(rng)
        fitness, loss_dict = calc_loss_population(_rng, params_ask)
        
        # Tell
        es_state = strategy.tell(params_ask, fitness, es_state, es_params)
        
        # Track
        best_idx = jnp.argmin(fitness)
        best_loss = float(fitness[best_idx])
        best_loss_prompt = float(loss_dict['loss_prompt'][best_idx])
        
        data_save['best_loss'].append(best_loss)
        data_save['mean_loss'].append(float(fitness.mean()))
        data_save['best_loss_prompt'].append(best_loss_prompt)
        data_save['best_params'].append(np.array(es_state.best_member))
        data_save['generation'].append(i_iter)
        
        # Save generation data
        gen_subdir = os.path.join(generations_dir, f"gen_{i_iter:04d}")
        os.makedirs(gen_subdir, exist_ok=True)
        
        gen_data = {
            'iteration': i_iter,
            'best_params': np.array(es_state.best_member),
            'best_fitness': float(es_state.best_fitness),
            'best_loss_prompt': best_loss_prompt,
            'mean': np.array(es_state.mean),
            'fitness_history': data_save['best_loss']
        }
        util.save_pkl(gen_subdir, "params", gen_data)
        
        # Render and save image
        rng, _rng = split(rng)
        img_overlay, _ = rollout_final_overlay_jit(_rng, es_state.best_member)
        
        # Save as PNG
        img_np = np.array(img_overlay * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(gen_subdir, "final_frame.png"))
        
        # Progress
        if (i_iter + 1) % 10 == 0:
            print(f"\nGen {i_iter}: Best Loss={best_loss:.4f} (Prompt={best_loss_prompt:.4f})")
        
        # Cleanup
        del img_overlay, fitness, loss_dict
        jax.clear_caches()
        gc.collect()
    
    # Save results
    util.save_pkl(args.save_dir, "best", (es_state.best_member, es_state.best_fitness))
    util.save_pkl(args.save_dir, "data_save", data_save)
    
    print("\n" + "="*80)
    print("Evolution Complete!")
    print("="*80)
    print(f"Final Best Loss: {es_state.best_fitness:.4f}")
    print(f"Saved to: {args.save_dir}")


if __name__ == '__main__':
    main(parse_args())

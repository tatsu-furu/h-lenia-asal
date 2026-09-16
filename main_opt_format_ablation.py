"""
ASAL Optimization: Input Format Ablation (2026-09-16)

experiment_overlay_hierarchical_microscope (JSAI submission's referenced run)
と同一設定(seed, rollout_steps, pop_size, n_iters, sigma, substrate params)を
保ったまま、CLIP入力画像の形式とCLIPモデルのみを変える。

--format で画像形式、--clip_model / --img_size でCLIPモデルを指定する。
5条件は起動コマンド側で作り分ける(本スクリプトはその共通実装)。

Base prompt text is never modified. --format に応じて、パネル説明文だけを
末尾に自動で追加する(three_panel / four_panel のみ)。
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from datetime import datetime
import gc
from PIL import Image

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import asal_metrics
import util

BASE_PROMPT = "Artificial microorganisms with a hierarchical structure consisting of multiple scales, as seen under a microscope"

PROMPT_ADDITIONS = {
    "grayscale": "",
    "rgb_overlay": "",
    "three_panel": (
        " The image is divided into three equal-width panels. From left to right: "
        "the smallest and fastest layer (L1), the intermediate layer (L2), and the "
        "largest and slowest layer (L3). Each panel is grayscale, where brighter "
        "pixels indicate higher activation."
    ),
    "four_panel": (
        " Four equal-width panels: grayscale L1, L2, L3 (finest to coarsest), "
        "brighter means higher activation; RGB composite (red=L1, green=L2, blue=L3), "
        "overlaps appear mixed or white."
    ),
}

FORMATS = list(PROMPT_ADDITIONS.keys())

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='h_lenia')
group.add_argument("--rollout_steps", type=int, default=500)

group = parser.add_argument_group("evaluation")
group.add_argument("--format", type=str, required=True, choices=FORMATS)
group.add_argument("--foundation_model", type=str, default="clip")
group.add_argument("--clip_model", type=str, required=True,
                    help="e.g. clip-vit-large-patch14 (224 native) or clip-vit-large-patch14-336 (336 native)")
group.add_argument("--img_size", type=int, default=336,
                    help="rendering resolution passed into render_state_format (not necessarily the CLIP model's own native size)")

group = parser.add_argument_group("optimization")
group.add_argument("--pop_size", type=int, default=30)
group.add_argument("--n_iters", type=int, default=100)
group.add_argument("--sigma", type=float, default=0.1)


def render_state_format(state, img_size, fmt):
    """Render H-Lenia state into a CLIP input image for the given format.

    grayscale / rgb_overlay: (img_size, img_size, 3)
    three_panel: (img_size, 3*img_size, 3)
    four_panel:  (img_size, 4*img_size, 3)
    """
    a1 = jnp.clip(jax.image.resize(jnp.mean(state["A1"], axis=-1), (img_size, img_size), method="linear"), 0.0, 1.0)
    a2 = jnp.clip(jax.image.resize(jnp.mean(state["A2"], axis=-1), (img_size, img_size), method="linear"), 0.0, 1.0)
    a3 = jnp.clip(jax.image.resize(jnp.mean(state["A3"], axis=-1), (img_size, img_size), method="linear"), 0.0, 1.0)

    if fmt == "grayscale":
        overlay = (a1 + a2 + a3) / 3.0
        return jnp.stack([overlay, overlay, overlay], axis=-1)
    elif fmt == "rgb_overlay":
        return jnp.stack([a1, a2, a3], axis=-1)
    elif fmt == "three_panel":
        p1, p2, p3 = (jnp.stack([x] * 3, axis=-1) for x in (a1, a2, a3))
        return jnp.concatenate([p1, p2, p3], axis=1)
    elif fmt == "four_panel":
        p1, p2, p3 = (jnp.stack([x] * 3, axis=-1) for x in (a1, a2, a3))
        rgb = jnp.stack([a1, a2, a3], axis=-1)
        return jnp.concatenate([p1, p2, p3, rgb], axis=1)
    else:
        raise ValueError(fmt)


def main(args):
    prompt_full = BASE_PROMPT + PROMPT_ADDITIONS[args.format]

    print("Configuration:")
    print(f"  Format: {args.format}")
    print(f"  CLIP model: {args.clip_model}")
    print(f"  img_size (render): {args.img_size}")
    print(f"  Prompt (full, actually used): {prompt_full!r}")
    print(f"  Seed: {args.seed}")
    print(f"  Population size: {args.pop_size}  Iterations: {args.n_iters}")

    from foundation_models.clip import CLIP
    fm = CLIP(clip_model=args.clip_model)
    print(f"  CLIP model native image_size (from processor): {fm.image_size}")

    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f'experiment_format_ablation_{args.format}_{timestamp}'

    os.makedirs(args.save_dir, exist_ok=True)
    generations_dir = os.path.join(args.save_dir, 'generations')
    os.makedirs(generations_dir, exist_ok=True)

    # 再発防止チェックリスト対応: 実際に使われた値(デフォルト依存せず)を
    # 全て明示的にargs.jsonへ書き出す。
    args_dict = vars(args).copy()
    args_dict['base_prompt'] = BASE_PROMPT
    args_dict['prompt_addition'] = PROMPT_ADDITIONS[args.format]
    args_dict['prompt_full_actually_used'] = prompt_full
    args_dict['clip_model_actually_used'] = args.clip_model
    args_dict['clip_native_image_size_from_processor'] = int(fm.image_size)
    util.save_json(args.save_dir, "args", args_dict)
    print(f"Saving to: {args.save_dir}")

    z_txt = fm.embed_txt([prompt_full])
    print(f"Text embedding shape: {z_txt.shape}")

    def rollout_final(rng, params):
        rng_init, rng_sim = split(rng)
        s0 = substrate.init_state(rng_init, params)

        def step_fn(state, _rng):
            return substrate.step_state(_rng, state, params), None

        state_final, _ = jax.lax.scan(step_fn, s0, split(rng_sim, args.rollout_steps))
        img = render_state_format(state_final, args.img_size, args.format)
        return img, state_final

    rollout_final_jit = jax.jit(rollout_final)

    def calc_loss_single(rng, params):
        img, state_final = rollout_final_jit(rng, params)
        z = fm.embed_img(img)
        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        loss_dict = dict(loss=loss_prompt, loss_prompt=loss_prompt)
        del img, state_final, z
        gc.collect()
        return loss_prompt, loss_dict

    calc_loss_jit = jax.jit(calc_loss_single)

    def calc_loss_population(rng, params_pop):
        losses, loss_dicts = [], []
        for i in range(params_pop.shape[0]):
            params_i = jax.tree_util.tree_map(lambda x: x[i], params_pop)
            rng, _rng = split(rng)
            loss, loss_dict = calc_loss_jit(_rng, params_i)
            losses.append(loss)
            loss_dicts.append(loss_dict)
            if (i + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()
        losses = jnp.array(losses)
        aggregated = {'loss': losses, 'loss_prompt': jnp.array([d['loss_prompt'] for d in loss_dicts])}
        return losses, aggregated

    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.num_params, elite_ratio=0.5)

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = split(rng)

    mean_init = (substrate.param_max + substrate.param_min) / 2.0
    es_params = strategy.default_params.replace(sigma_init=args.sigma)
    es_state = strategy.initialize(_rng, es_params)
    es_state = es_state.replace(mean=mean_init)

    data_save = {'best_loss': [], 'mean_loss': [], 'best_loss_prompt': [], 'best_params': [], 'generation': []}

    print("\n" + "=" * 80)
    print(f"Starting Evolution (format={args.format}, clip_model={args.clip_model})")
    print("=" * 80)

    for i_iter in tqdm(range(args.n_iters), desc="Evolution"):
        rng, _rng = split(rng)
        params_ask, es_state = strategy.ask(_rng, es_state, es_params)
        params_ask = jnp.clip(params_ask, substrate.param_min, substrate.param_max)

        rng, _rng = split(rng)
        fitness, loss_dict = calc_loss_population(_rng, params_ask)

        es_state = strategy.tell(params_ask, fitness, es_state, es_params)

        best_idx = jnp.argmin(fitness)
        best_loss = float(fitness[best_idx])
        best_loss_prompt = float(loss_dict['loss_prompt'][best_idx])

        data_save['best_loss'].append(best_loss)
        data_save['mean_loss'].append(float(fitness.mean()))
        data_save['best_loss_prompt'].append(best_loss_prompt)
        data_save['best_params'].append(np.array(es_state.best_member))
        data_save['generation'].append(i_iter)

        gen_subdir = os.path.join(generations_dir, f"gen_{i_iter:04d}")
        os.makedirs(gen_subdir, exist_ok=True)
        gen_data = {
            'iteration': i_iter,
            'best_params': np.array(es_state.best_member),
            'best_fitness': float(es_state.best_fitness),
            'best_loss_prompt': best_loss_prompt,
            'mean': np.array(es_state.mean),
            'fitness_history': data_save['best_loss'],
        }
        util.save_pkl(gen_subdir, "params", gen_data)

        rng, _rng = split(rng)
        img, _ = rollout_final_jit(_rng, es_state.best_member)
        img_np = np.array(img * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(gen_subdir, "final_frame.png"))

        # checkpoint: 中断・再開に使えるよう毎世代 data_save を上書き保存
        util.save_pkl(args.save_dir, "data_save", data_save)
        util.save_pkl(args.save_dir, "best", (es_state.best_member, es_state.best_fitness))

        if (i_iter + 1) % 10 == 0 or i_iter == 0:
            print(f"\nGen {i_iter}: Best Loss={best_loss:.4f} (Prompt={best_loss_prompt:.4f})")

        del img, fitness, loss_dict
        jax.clear_caches()
        gc.collect()

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print(f"Final Best Loss: {es_state.best_fitness:.4f}")
    print(f"Saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main(parser.parse_args())

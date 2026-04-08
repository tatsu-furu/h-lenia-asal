# -*- coding: utf-8 -*-
"""
H-Lenia シミュレーションステップ実行関数 (v59 - JIT最適化対応 最終版)
- main.py側でのJITコンパイル最適化に対応するため、kの値を動的に受け取るように修正
"""
import jax, jax.numpy as jnp, functools
from .h_lenia_lenia import _lenia_update_step
from .h_lenia_interaction import compute_interaction

def _simulation_step(
    # --- 各ステップで変化する引数 ---
    state,
    k_strengths, # k_12, k_21.. ではなく、(0.5, 0.5, ...) のようなタプルで受け取る
    _,           # jax.lax.scan で使うためのプレースホルダー

    # --- シミュレーション中に変化しない「静的な」引数 ---
    simulate_layer1, simulate_layer2, simulate_layer3,
    type_1to2, type_2to1, type_2to3, type_3to2,
    time_scale_1, time_scale_2, time_scale_3,
    repulsion_mode_is_1_minus_abs,
    downsample_method,
    nc1, nc2, nc3,
    
    # --- シミュレーション中に変化しない「動的な」配列引数 ---
    fK1, m1, s1, h1, c01, c11, T1,
    fK2, m2, s2, h2, c02, c12, T2,
    fK3, m3, s3, h3, c03, c13, T3
):
    """H-Leniaモデルを1ステップ分進める関数"""
    k_12, k_21, k_23, k_32 = k_strengths

    frame = state["frame"]
    A1_curr, A2_curr, A3_curr = state["A1"], state["A2"], state["A3"]
    disp_inter_L2_to_L1_prev = state["display_inter_L2_to_L1"]
    disp_inter_L1_to_L2_prev = state["display_inter_L1_to_L2"]
    disp_inter_L3_to_L2_prev = state["display_inter_L3_to_L2"]
    disp_inter_L2_to_L3_prev = state["display_inter_L2_to_L3"]

    A1_updated = jax.lax.cond(
        simulate_layer1 and (jnp.mod(frame, time_scale_1) == 0),
        lambda: _lenia_update_step(A1_curr, fK1, m1, s1, h1, c01, c11, T1, nc1),
        lambda: A1_curr
    )
    A2_updated = jax.lax.cond(
        simulate_layer2 and (jnp.mod(frame, time_scale_2) == 0),
        lambda: _lenia_update_step(A2_curr, fK2, m2, s2, h2, c02, c12, T2, nc2),
        lambda: A2_curr
    )
    A3_updated = jax.lax.cond(
        simulate_layer3 and (jnp.mod(frame, time_scale_3) == 0),
        lambda: _lenia_update_step(A3_curr, fK3, m3, s3, h3, c03, c13, T3, nc3),
        lambda: A3_curr
    )
    
    def compute_interaction_force(source, target, k_val, type_val, repulsion_flag, downsample_flag):
        force = compute_interaction(source, target, k_val, type_val, repulsion_flag, downsample_flag)
        return force, force

    interaction_A1, disp_inter_L2_to_L1 = jax.lax.cond(
        simulate_layer1 and (type_2to1 != 2) and (jnp.mod(frame, time_scale_1) == 0),
        lambda: compute_interaction_force(A2_updated, A1_updated, k_21, type_2to1, repulsion_mode_is_1_minus_abs, downsample_method),
        lambda: (jnp.zeros_like(A1_updated), disp_inter_L2_to_L1_prev)
    )
    interaction_A2_from_L1, disp_inter_L1_to_L2 = jax.lax.cond(
        simulate_layer2 and (type_1to2 != 2) and (jnp.mod(frame, time_scale_2) == 0),
        lambda: compute_interaction_force(A1_updated, A2_updated, k_12, type_1to2, repulsion_mode_is_1_minus_abs, downsample_method),
        lambda: (jnp.zeros_like(A2_updated), disp_inter_L1_to_L2_prev)
    )
    interaction_A2_from_L3, disp_inter_L3_to_L2 = jax.lax.cond(
        simulate_layer2 and (type_3to2 != 2) and (jnp.mod(frame, time_scale_2) == 0),
        lambda: compute_interaction_force(A3_updated, A2_updated, k_32, type_3to2, repulsion_mode_is_1_minus_abs, downsample_method),
        lambda: (jnp.zeros_like(A2_updated), disp_inter_L3_to_L2_prev)
    )
    interaction_A3, disp_inter_L2_to_L3 = jax.lax.cond(
        simulate_layer3 and (type_2to3 != 2) and (jnp.mod(frame, time_scale_3) == 0),
        lambda: compute_interaction_force(A2_updated, A3_updated, k_23, type_2to3, repulsion_mode_is_1_minus_abs, downsample_method),
        lambda: (jnp.zeros_like(A3_updated), disp_inter_L2_to_L3_prev)
    )
    
    A1_new = jnp.clip(A1_updated + interaction_A1, 0.0, 1.0)
    A2_new = jnp.clip(A2_updated + interaction_A2_from_L1 + interaction_A2_from_L3, 0.0, 1.0)
    A3_new = jnp.clip(A3_updated + interaction_A3, 0.0, 1.0)
    
    new_state = {
        "A1": A1_new.astype(jnp.float32), 
        "A2": A2_new.astype(jnp.float32), 
        "A3": A3_new.astype(jnp.float32),
        "frame": frame + 1,
        "display_inter_L2_to_L1": disp_inter_L2_to_L1.astype(jnp.float32),
        "display_inter_L1_to_L2": disp_inter_L1_to_L2.astype(jnp.float32),
        "display_inter_L3_to_L2": disp_inter_L3_to_L2.astype(jnp.float32),
        "display_inter_L2_to_L3": disp_inter_L2_to_L3.astype(jnp.float32),
        "display_inter_L2_total": (interaction_A2_from_L1 + interaction_A2_from_L3).astype(jnp.float32),
    }
    return new_state, None
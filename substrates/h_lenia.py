import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import re

from . import h_lenia_simulator as sim
from . import h_lenia_lenia as lenia
from . import pattern
from .h_lenia_config import SimulationConfig

class HLenia:
    """
    Hierarchical Lenia (H-Lenia) substrate for ASAL.
    Optimizes the growth parameters (m, s) offsets for a fixed Aquarium pattern structure.
    """
    def __init__(self):
        self.config = SimulationConfig()
        self.rollout_steps = self.config.n_anim * self.config.steps_per_frame

        self.param_min = jnp.array([-0.1, -0.01, -0.1, -0.01, -0.1, -0.01])
        self.param_max = jnp.array([ 0.1,  0.01,  0.1,  0.01,  0.1,  0.01])
        self.num_params = 6

        # Thesis Config
        self.thesis_config = "R(0.6)/A(0.5)R(0.9)/A(0.4)"
        self.k_strengths, self.inter_types = self._parse_thesis_notation(self.thesis_config)
        
        print(f"H-Lenia Configured from notation: {self.thesis_config}")

    def _parse_thesis_notation(self, notation):
        type_map = {'A': 0, 'R': 1, 'N': 2}
        parts = notation.split('/')
        if len(parts) != 3: raise ValueError(f"Invalid notation format: {notation}")
        pattern = re.compile(r"([ARN])\((\d+\.?\d*)\)")
        
        matches_top = pattern.findall(parts[0])
        char_23, val_23 = matches_top[0]
        k23, t23 = float(val_23), type_map[char_23]

        matches_mid = pattern.findall(parts[1])
        char_32, val_32 = matches_mid[0]
        k32, t32 = float(val_32), type_map[char_32]
        char_12, val_12 = matches_mid[1]
        k12, t12 = float(val_12), type_map[char_12]

        matches_bot = pattern.findall(parts[2])
        char_21, val_21 = matches_bot[0]
        k21, t21 = float(val_21), type_map[char_21]

        return (k12, k21, k23, k32), (t12, t21, t23, t32)

    def default_params(self, rng):
        return random.uniform(rng, (self.num_params,), minval=self.param_min, maxval=self.param_max)

    def _get_modified_pattern(self, base_pattern_name, dm, ds):
        base_pat = pattern.pattern.get(base_pattern_name, pattern.pattern.get('aquarium'))
        new_pat = base_pat.copy()
        new_kernels = []
        kernels = base_pat.get("kernels", [])
        if not kernels and "b" in base_pat: 
            kernels = [{"b": base_pat["b"], "m": base_pat.get("m"), "s": base_pat.get("s")}]
        for k in kernels:
            new_k = k.copy()
            if "m" in new_k: new_k["m"] = new_k["m"] + dm
            if "s" in new_k: new_k["s"] = jnp.maximum(0.001, new_k["s"] + ds)
            new_kernels.append(new_k)
        new_pat["kernels"] = new_kernels
        return new_pat

    def init_state(self, rng, params):
        key1, key2, key3 = random.split(rng, 3)
        p1_name = getattr(self.config, 'pattern_name1', 'aquarium')
        p2_name = getattr(self.config, 'pattern_name2', 'aquarium')
        p3_name = getattr(self.config, 'pattern_name3', 'aquarium')
        
        p1 = self._get_modified_pattern(p1_name, params[0], params[1])
        p2 = self._get_modified_pattern(p2_name, params[2], params[3])
        p3 = self._get_modified_pattern(p3_name, params[4], params[5])

        A1, _, _, _, _, _, _, _, _ = lenia.initialize_world(p1, self.config.size_1, 1, self.config.num_individuals_1, key1)
        A2, _, _, _, _, _, _, _, _ = lenia.initialize_world(p2, self.config.size_2, 1, 0, key2)
        A3, _, _, _, _, _, _, _, _ = lenia.initialize_world(p3, self.config.size_3, 1, 0, key3)

        state = {
            "A1": A1.astype(jnp.float32),
            "A2": A2.astype(jnp.float32),
            "A3": A3.astype(jnp.float32),
            "frame": jnp.array(0, dtype=jnp.int32),
            "display_inter_L2_to_L1": jnp.zeros_like(A1),
            "display_inter_L1_to_L2": jnp.zeros_like(A2),
            "display_inter_L3_to_L2": jnp.zeros_like(A2),
            "display_inter_L2_to_L3": jnp.zeros_like(A3),
            "display_inter_L2_total": jnp.zeros_like(A2),
        }
        return state

    def step_state(self, rng, state, params):
        p1_name = getattr(self.config, 'pattern_name1', 'aquarium')
        p2_name = getattr(self.config, 'pattern_name2', 'aquarium')
        p3_name = getattr(self.config, 'pattern_name3', 'aquarium')
        
        p1 = self._get_modified_pattern(p1_name, params[0], params[1])
        p2 = self._get_modified_pattern(p2_name, params[2], params[3])
        p3 = self._get_modified_pattern(p3_name, params[4], params[5])

        dummy_key = random.PRNGKey(0)
        _, m1, s1, h1, c01, c11, R1, nc1, T1 = lenia.initialize_world(p1, self.config.size_1, 1, 0, dummy_key)
        fK1 = lenia.compute_kernel(p1, R1, self.config.size_1)
        _, m2, s2, h2, c02, c12, R2, nc2, T2 = lenia.initialize_world(p2, self.config.size_2, 1, 0, dummy_key)
        fK2 = lenia.compute_kernel(p2, R2, self.config.size_2)
        _, m3, s3, h3, c03, c13, R3, nc3, T3 = lenia.initialize_world(p3, self.config.size_3, 1, 0, dummy_key)
        fK3 = lenia.compute_kernel(p3, R3, self.config.size_3)

        t12, t21, t23, t32 = self.inter_types

        new_state, _ = sim._simulation_step(
            state, self.k_strengths, None,
            True, True, True,
            t12, t21, t23, t32,
            self.config.time_scale_1, self.config.time_scale_2, self.config.time_scale_3,
            (self.config.repulsion_mode == '1_minus_abs'),
            self.config.downsample_method,
            nc1, nc2, nc3,
            fK1, m1, s1, h1, c01, c11, T1,
            fK2, m2, s2, h2, c02, c12, T2,
            fK3, m3, s3, h3, c03, c13, T3
        )
        return new_state

    def render_state(self, state, params, img_size=None):
        """
        現在の状態を画像としてレンダリングする。
        【メモリ対策】
        GPUメモリ爆発を防ぐため、外部からのサイズ指定を無視し、
        CLIP評価用サイズ (224px) に強制リサイズして返す。
        """
        target_size = 224
        
        a1_density = jnp.mean(state["A1"], axis=-1)
        a2_density = jnp.mean(state["A2"], axis=-1)
        a3_density = jnp.mean(state["A3"], axis=-1)

        img_r = jax.image.resize(a1_density, (target_size, target_size), method="linear")
        img_g = jax.image.resize(a2_density, (target_size, target_size), method="linear")
        img_b = jax.image.resize(a3_density, (target_size, target_size), method="linear")
        
        return jnp.clip(jnp.stack([img_r, img_g, img_b], axis=-1), 0.0, 1.0)
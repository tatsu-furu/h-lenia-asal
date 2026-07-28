
import jax
import jax.numpy as jnp
from jax.random import split
from matplotlib import colors as mcolors
from .gol import GameOfLife

"""
This is the code for simulating the competition between different game of life rules as shown in https://sakana.ai/asal/.
"""
class GameOfLifeCompeting():
    def __init__(self, k_sims=4, grid_size=128, double_step=True,
                 colors='448aff-1565c0-009688-8bc34a-ffc107-ff9800-f44336-ad1457-448aff-1565c0-009688-8bc34a-ffc107-ff9800-f44336-ad1457'):
        assert k_sims == 4 or k_sims == 9 or k_sims == 16 or k_sims == 25 or k_sims == 36
        self.k_sims = k_sims
        self.sqrt_ksims = int(jnp.sqrt(k_sims))
        self.sim = GameOfLife(grid_size=grid_size)
        assert self.k_sims <= len(colors.split('-'))
        self.species_colors = jnp.array([mcolors.to_rgb(f"#{c}") for c in colors.split('-')])[:self.k_sims]
        self.double_step = double_step

    def default_params(self, rng):
        return jax.vmap(self.model_boids.default_params, in_axes=(0,))(split(rng, self.k_sims))

    def init_state(self, rng, params):
        state = self.sim.init_state(rng, params[0])
        rule_state = jnp.arange(self.k_sims).reshape(self.sqrt_ksims, self.sqrt_ksims)
        rule_state = repeat(rule_state, "x y -> (x W) (y H)", W=self.sim.grid_size//self.sqrt_ksims, H=self.sim.grid_size//self.sqrt_ksims)
        return dict(state=state, rule_state=rule_state)

    def step_state(self, rng, state, params):
        state, rule_state = state['state'], state['rule_state']

        def step_fn(rng, state, params):
            state = self.sim.step_state(rng, state, params)
            if self.double_step:
                state = self.sim.step_state(rng, state, params)
            return state
        state = jax.vmap(step_fn, in_axes=(None, None, 0))(rng, state, params)
        state = rearrange(state, "D H W -> H W D")
        def index_fn(states, rule_idx):
            return states[rule_idx]
        state = jax.vmap(jax.vmap(index_fn))(state, rule_state)

        # CHANGING DYNANMICS CODE
        def get_neighbors(x):
            x = jnp.pad(x, pad_width=1, mode='wrap')
            neighs = jnp.stack([x[:-2, :-2], x[:-2, 1:-1], x[:-2, 2:], x[1:-1, :-2], x[1:-1, 2:], x[2:, :-2], x[2:, 1:-1], x[2:, 2:]], axis=-1)
            return neighs
        state_neighs = get_neighbors(state)
        rule_state_neighs = get_neighbors(rule_state)

        def get_rule_idx(rng, state, rule_state, state_neighs, rule_state_neighs):
            state_neighs = jax.random.permutation(rng, state_neighs)
            rule_state_neighs = jax.random.permutation(rng, rule_state_neighs)

            rule_state_2 = rule_state_neighs[jnp.argmax(state_neighs)]
            # only change rule_state if state is dead and there is a living neighbor
            return jax.lax.select((state==0)& (state_neighs.sum()>0), rule_state_2, rule_state)

        state_neighs = rearrange(state_neighs, "H W D -> (H W) D")
        rule_state_neighs = rearrange(rule_state_neighs, "H W D -> (H W) D")
        rule_state = jax.vmap(get_rule_idx)(split(rng, len(state_neighs)), state.flatten(), rule_state.flatten(), state_neighs, rule_state_neighs)
        rule_state = rule_state.reshape(*state.shape)
        return dict(state=state, rule_state=rule_state)
    
    def render_state(self, state, params, img_size=None):
        state, rule_state = state['state'], state['rule_state']
        img = repeat(state.astype(float), "H W -> H W 3")
        img = img * self.species_colors[rule_state]
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img

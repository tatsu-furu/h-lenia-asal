import jax
import jax.numpy as jnp
from einops import rearrange

class Pixels():
    def __init__(self):
        self.img_mean = jnp.array([0.485, 0.456, 0.406]) # from the dino preprocessor
        self.img_std = jnp.array([0.229, 0.224, 0.225])

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 C H W")
        z_img = rearrange(img, "1 C (H h) (W w) -> (C H W) (h w)", h=8, w=8).mean(axis=-1)
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True) # normalizing isn't too good but should be fine
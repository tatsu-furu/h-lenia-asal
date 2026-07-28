import jax
import jax.numpy as jnp
from einops import rearrange
from transformers import AutoImageProcessor, FlaxDinov2Model

class DINO():
    def __init__(self, dino_model="dinov2-base", features='pooler'):
        self.processor = AutoImageProcessor.from_pretrained(f"facebook/{dino_model}")
        self.dino_model = FlaxDinov2Model.from_pretrained(f"facebook/{dino_model}")
        self.features = features

        self.img_mean = jnp.array(self.processor.image_mean)
        self.img_std = jnp.array(self.processor.image_std)

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 C H W")
        outputs = self.dino_model(pixel_values=img)
        if self.features == 'pooler':
            z_img = outputs.pooler_output[0]
        elif self.features == 'avg_pool':
            z_img = outputs.last_hidden_state[0, :].mean(axis=0)
        else:
            raise ValueError(f"features {self.features} not recognized")
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True) # normalizing isn't ideal but should be fine
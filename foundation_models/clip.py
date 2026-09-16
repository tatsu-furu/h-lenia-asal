# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
from einops import rearrange
from transformers import AutoProcessor, FlaxCLIPModel

class CLIP():
    def __init__(self, clip_model="clip-vit-base-patch32"):
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        try:
            self.clip_model = FlaxCLIPModel.from_pretrained(f"openai/{clip_model}")
        except OSError:
            # openai/clip-vit-large-patch14-336 等、Flax重み(flax_model.msgpack)が
            # HF Hubに存在しないモデルはPyTorch重みから変換してロードする
            # (2026-09-16判明: 336版はFlax重み非公開)
            self.clip_model = FlaxCLIPModel.from_pretrained(f"openai/{clip_model}", from_pt=True)

        self.img_mean = jnp.array(self.processor.image_processor.image_mean)
        self.img_std = jnp.array(self.processor.image_processor.image_std)
        
        # Determine supported image size from model config
        # ViT-B/32 and ViT-B/16 use 224, ViT-L/14 uses 336
        self.image_size = self.processor.image_processor.size.get('shortest_edge', 224)

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        
        Supports 224x224 (ViT-B) or 336x336 (ViT-L) depending on model.
        """
        H, W, C = img.shape
        target_size = self.image_size
        
        if H != target_size or W != target_size:
            img = jax.image.resize(img, (target_size, target_size, C), method='bilinear')
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 C H W")
        z_img = self.clip_model.get_image_features(img)[0]
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)

    def embed_txt(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        inputs = self.processor(text=prompts, return_tensors="jax", padding=True)
        z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return z_text / jnp.linalg.norm(z_text, axis=-1, keepdims=True)
    
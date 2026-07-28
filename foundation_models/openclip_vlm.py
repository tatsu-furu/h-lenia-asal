"""
OpenCLIP ViT-H/14 wrapper for GPU0 (RTX 5070 Ti).

GPU分担:
  GPU0 (RTX 5070 Ti, 16GB): OpenCLIP 埋め込み (~5GB)
  GPU1 (RTX 3090, 24GB):    JAX H-Lenia シミュレーション

入出力はすべて numpy (CPU) で JAX ↔ PyTorch を安全にブリッジ。
"""
import numpy as np
import torch
from PIL import Image
from typing import List


class OpenCLIPVLM:
    """
    OpenCLIP ViT-H/14 (laion2b) on GPU0 (RTX 5070 Ti).
    embed_dim = 1024, VRAM ~5GB.
    """

    def __init__(self, device: str = "cuda:0"):  # RTX 3090 (CUDA device 0, 24GB)
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip not installed.\n"
                "Run: pip install open-clip-torch"
            )
        self.device = torch.device(device)
        arch, pretrained = "ViT-H-14", "laion2b_s32b_b79k"
        print(f"[OpenCLIP] Loading {arch} ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model.eval()
        self.embed_dim = self.model.visual.output_dim  # 1024
        print(f"[OpenCLIP] Ready. embed_dim={self.embed_dim}")

    def _l2(self, t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def embed_img(self, img: np.ndarray) -> np.ndarray:
        """img: (H, W, 3) float32 [0,1] → (D,) float32"""
        return self.embed_img_batch(img[None])[0]

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        """imgs: (K, H, W, 3) float32 [0,1] → (K, D) float32"""
        imgs_u8 = (imgs * 255).clip(0, 255).astype(np.uint8)
        pil_imgs = [Image.fromarray(img) for img in imgs_u8]
        tensors = torch.stack([self.preprocess(img) for img in pil_imgs]).to(self.device)
        feat = self.model.encode_image(tensors)
        feat = self._l2(feat)
        return feat.cpu().float().numpy()

    @torch.no_grad()
    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        """prompts: list of str → (P, D) float32"""
        tokens = self.tokenizer(prompts).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = self._l2(feat)
        return feat.cpu().float().numpy()

    def vram_used_mb(self) -> int:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) // (1024 ** 2)
        return 0

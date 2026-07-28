"""
VLM wrapper for GPU1 (PyTorch) in dual-GPU ASAL setup.

GPU0 (RTX 5070 Ti, 16GB): JAX / H-Lenia simulation
GPU1 (RTX 3090, 24GB):    PyTorch / VLM image embedding + scoring

All inputs/outputs are numpy arrays (CPU) to bridge JAX ↔ PyTorch safely.
"""

import numpy as np
import torch
from PIL import Image
from typing import List


# ── Base class ────────────────────────────────────────────────────────────────

class VLMBase:
    """Abstract base for all VLM wrappers. Inputs/outputs are numpy arrays."""

    def __init__(self, device: str = "cuda:1"):
        self.device = torch.device(device)
        self.embed_dim: int = 0  # set by subclass

    @torch.no_grad()
    def embed_img(self, img: np.ndarray) -> np.ndarray:
        """Single image. img: (H, W, 3) float32 [0,1] → (D,) float32, L2-normed."""
        return self.embed_img_batch(img[None])[0]

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        """Batch of images. imgs: (K, H, W, 3) float32 [0,1] → (K, D) float32, L2-normed."""
        raise NotImplementedError

    @torch.no_grad()
    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        """prompts: list of strings → (P, D) float32, L2-normed."""
        raise NotImplementedError

    def _to_pil(self, imgs: np.ndarray) -> List[Image.Image]:
        """(K, H, W, 3) float32 [0,1] → list of PIL Images."""
        imgs_u8 = (imgs * 255).clip(0, 255).astype(np.uint8)
        return [Image.fromarray(img) for img in imgs_u8]

    def _l2_norm(self, t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device}, embed_dim={self.embed_dim})"


# ── CLIP Large (default) ──────────────────────────────────────────────────────

class CLIPLarge(VLMBase):
    """
    CLIP via openai/clip パッケージ (pip install clip).
    ViT-L/14@336px: embed_dim=768, ~1.7 GB VRAM.
    ネットワーク不要・ローカルキャッシュ利用。
    """

    CLIP_NAMES = {
        "clip_b32":      "ViT-B/32",
        "clip_l14":      "ViT-L/14",
        "clip_l14_336":  "ViT-L/14@336px",
    }

    def __init__(self, model_name: str = "clip_l14_336", device: str = "cuda:1"):
        super().__init__(device)
        import clip as openai_clip
        clip_name = self.CLIP_NAMES.get(model_name, "ViT-L/14@336px")
        print(f"[VLM] Loading CLIP {clip_name} on {device}...")
        self.model, self.preprocess = openai_clip.load(clip_name, device=self.device)
        self.model.eval()
        import clip as _c
        self._clip = _c
        self.embed_dim = self.model.visual.output_dim  # 768 for L/14

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        pil_imgs = self._to_pil(imgs)
        tensors = torch.stack([self.preprocess(img) for img in pil_imgs]).to(self.device)
        feat = self.model.encode_image(tensors).float()
        feat = self._l2_norm(feat)
        return feat.cpu().numpy()

    @torch.no_grad()
    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        tokens = self._clip.tokenize(prompts).to(self.device)
        feat = self.model.encode_text(tokens).float()
        feat = self._l2_norm(feat)
        return feat.cpu().numpy()


# ── OpenCLIP (ViT-H/14 or ViT-G/14) ─────────────────────────────────────────

class OpenCLIPModel(VLMBase):
    """
    OpenCLIP via open-clip-torch (pip install open-clip-torch).
    ViT-H-14 laion2b: ~5 GB VRAM, embed_dim=1024.
    ViT-G-14 laion2b: ~7 GB VRAM, embed_dim=1280.
    """

    CONFIGS = {
        "openclip_h14": ("ViT-H-14",  "laion2b_s32b_b79k"),
        "openclip_g14": ("ViT-G-14",  "laion2b_s34b_b88k"),
        "openclip_l14": ("ViT-L-14",  "laion2b_s32b_b82k"),
    }

    def __init__(self, model_name: str = "openclip_h14", device: str = "cuda:1"):
        super().__init__(device)
        try:
            import open_clip
        except ImportError:
            raise ImportError("Install open_clip: pip install open-clip-torch")

        arch, pretrained = self.CONFIGS.get(model_name, ("ViT-H-14", "laion2b_s32b_b79k"))
        print(f"[VLM/GPU1] Loading OpenCLIP {arch} ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model.eval()
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        pil_imgs = self._to_pil(imgs)
        tensors = torch.stack([self.preprocess(img) for img in pil_imgs]).to(self.device)
        feat = self.model.encode_image(tensors)
        feat = self._l2_norm(feat)
        return feat.cpu().float().numpy()

    @torch.no_grad()
    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = self._l2_norm(feat)
        return feat.cpu().float().numpy()


# ── LLaVA (vision encoder only) ──────────────────────────────────────────────

class LLaVAVision(VLMBase):
    """
    LLaVA-1.5 vision tower (CLIP ViT-L/14@336) with 4-bit quantization.
    Extracts patch embeddings from the penultimate vision layer.
    embed_dim = 1024, ~4-6 GB VRAM (4-bit).

    Note:
    - embed_txt() is NOT supported (LLaVA has no standalone text encoder).
      Use clip_l14_336 or openclip_h14 for prompt-based scoring with LLaVA.
    - bitsandbytes 4-bit may require Linux/WSL2. See troubleshooting below.

    Troubleshooting (Windows):
    - If bitsandbytes fails: use llava_7b_fp16 (no quantization) or switch to WSL2.
    - Alternative: pip install bitsandbytes-windows
    """

    HF_NAMES = {
        "llava_7b":  "llava-hf/llava-1.5-7b-hf",
        "llava_13b": "llava-hf/llava-1.5-13b-hf",
    }

    def __init__(self, model_name: str = "llava_7b", device: str = "cuda:1"):
        super().__init__(device)
        hf_name = self.HF_NAMES.get(model_name, "llava-hf/llava-1.5-7b-hf")

        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print(f"[VLM/GPU1] Loading LLaVA {hf_name} (4-bit) on {device}...")
            full_model = LlavaForConditionalGeneration.from_pretrained(
                hf_name,
                quantization_config=bnb_config,
                device_map={"": str(self.device)},
            )
            self.processor = AutoProcessor.from_pretrained(hf_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLaVA with 4-bit quantization: {e}\n"
                "Try: pip install bitsandbytes  (or bitsandbytes-windows on Windows)"
            )

        # Keep only the vision tower to save memory
        self.vision_tower = full_model.model.vision_tower.eval()
        self.image_processor = self.processor.image_processor
        del full_model
        torch.cuda.empty_cache()

        self.embed_dim = self.vision_tower.config.hidden_size  # 1024 for ViT-L/14

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        pil_imgs = self._to_pil(imgs)
        # LLaVA uses 336px CLIP image processor
        inputs = self.image_processor(images=pil_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)

        # Extract penultimate layer features (output_hidden_states=True)
        outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # hidden_states[-2]: (B, N_patches+1, D), exclude CLS token [1:]
        feat = outputs.hidden_states[-2][:, 1:, :].mean(dim=1).float()  # (B, D)
        feat = self._l2_norm(feat)
        return feat.cpu().numpy()

    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "LLaVA has no standalone text encoder.\n"
            "Use --vlm_model clip_l14_336 or openclip_h14 for text prompt scoring.\n"
            "Or set --coef_prompt 0.0 to use OE score only."
        )


# ── SigLIP (Google, strong CLIP alternative) ──────────────────────────────────

class SigLIPModel(VLMBase):
    """
    SigLIP (Sigmoid Loss for Language Image Pre-training).
    google/siglip-so400m-patch14-384: embed_dim=1152, ~3 GB VRAM.
    Better than CLIP for zero-shot on many benchmarks.
    """

    HF_NAMES = {
        "siglip_so400m": "google/siglip-so400m-patch14-384",
        "siglip_base":   "google/siglip-base-patch16-224",
        "siglip_large":  "google/siglip-large-patch16-256",
    }

    def __init__(self, model_name: str = "siglip_so400m", device: str = "cuda:1"):
        super().__init__(device)
        from transformers import AutoModel, AutoProcessor
        hf_name = self.HF_NAMES.get(model_name, model_name)
        print(f"[VLM/GPU1] Loading SigLIP {hf_name} on {device}...")
        self.model = AutoModel.from_pretrained(hf_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_name)
        self.embed_dim = self.model.config.vision_config.hidden_size

    @torch.no_grad()
    def embed_img_batch(self, imgs: np.ndarray) -> np.ndarray:
        pil_imgs = self._to_pil(imgs)
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_image_features(**inputs)
        feat = self._l2_norm(feat)
        return feat.cpu().float().numpy()

    @torch.no_grad()
    def embed_txt(self, prompts: List[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, return_tensors="pt", padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_text_features(**inputs)
        feat = self._l2_norm(feat)
        return feat.cpu().float().numpy()


# ── Factory ───────────────────────────────────────────────────────────────────

_MODEL_MAP = {
    "clip_b32":       (CLIPLarge,     {}),
    "clip_l14":       (CLIPLarge,     {}),
    "clip_l14_336":   (CLIPLarge,     {}),
    "openclip_h14":   (OpenCLIPModel, {}),
    "openclip_g14":   (OpenCLIPModel, {}),
    "openclip_l14":   (OpenCLIPModel, {}),
    "llava_7b":       (LLaVAVision,   {}),
    "llava_13b":      (LLaVAVision,   {}),
    "siglip_so400m":  (SigLIPModel,   {}),
    "siglip_base":    (SigLIPModel,   {}),
    "siglip_large":   (SigLIPModel,   {}),
}


def create_vlm(model_name: str, device: str = "cuda:1") -> VLMBase:
    """
    Create a VLM model for GPU1.

    Supported model_name:
      CLIP:      clip_b32, clip_l14, clip_l14_336 (default)
      OpenCLIP:  openclip_h14, openclip_g14, openclip_l14
      LLaVA:     llava_7b, llava_13b
      SigLIP:    siglip_so400m, siglip_base, siglip_large
    """
    if model_name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown VLM: '{model_name}'. "
            f"Available: {list(_MODEL_MAP.keys())}"
        )
    cls, _ = _MODEL_MAP[model_name]
    return cls(model_name=model_name, device=device)

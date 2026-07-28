"""
LLaVA-1.5-7B ジャッジ (RTX 3090, cuda:0)

H-Lenia シミュレーションフレームを見て「生物パターンとして面白いか」を
1-10 スコアで返す。OpenCLIP と同じ GPU に 4bit 量子化で同居。

VRAM: OpenCLIP(~5GB) + LLaVA-7B-4bit(~7GB) ≈ 12GB < 24GB (RTX 3090)
"""
import re
import json
import os as _os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image


JUDGE_PROMPT = (
    "You are evaluating a simulation of artificial life (Lenia cellular automaton).\n"
    "Look at this frame and rate it from 1 to 10 based on these criteria:\n"
    "  1-2: Empty, black screen, or uniform noise — no structure\n"
    "  3-4: Minimal activity, cells dying out or barely moving\n"
    "  5-6: Some structure visible but simple or static\n"
    "  7-8: Clear spatial patterns, dynamic movement, organism-like behavior\n"
    "  9-10: Rich complex patterns, multiple interacting structures, life-like\n"
    "Reply with ONLY a single integer from 1 to 10."
)


class LLaVAJudge:
    """
    LLaVA-1.5-7B を使ってシミュレーションフレームを評価する。
    スコア 1-10 を返す (高いほど良い)。
    """

    def __init__(self, device: str = "cuda:0",
                 model_id: str = "llava-hf/llava-1.5-7b-hf"):
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        except ImportError:
            raise ImportError("pip install transformers bitsandbytes accelerate")

        print(f"[LLaVA] Loading {model_id} on {device} (4-bit)...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        # processor_config.json に patch_size を書き込んで再ロード
        # (transformers 4.46+ は属性直接設定が効かず json から読む)
        patch_size = self.model.config.vision_config.patch_size  # 通常14
        tmpdir = tempfile.mkdtemp()
        try:
            base_proc = AutoProcessor.from_pretrained(model_id)
            base_proc.save_pretrained(tmpdir)
            proc_cfg = {
                "patch_size": patch_size,
                "vision_feature_select_strategy": "default",
                "processor_class": "LlavaProcessor",
            }
            with open(_os.path.join(tmpdir, "processor_config.json"), "w") as f:
                json.dump(proc_cfg, f)
            self.processor = AutoProcessor.from_pretrained(tmpdir)
        finally:
            shutil.rmtree(tmpdir)
        self.device = device
        self.model.eval()
        print(f"[LLaVA] Ready. (patch_size={self.processor.patch_size})")

    @torch.no_grad()
    def score_frame(self, img: np.ndarray) -> float:
        """
        img: (H, W, 3) float32 [0,1]
        returns: float score 1.0-10.0
        """
        pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        # LLaVA-1.5 の正式フォーマット: "USER: <image>\n{text}\nASSISTANT:"
        prompt = f"USER: <image>\n{JUDGE_PROMPT}\nASSISTANT:"
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = self.processor.decode(output[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()
        return self._parse_score(generated)

    @torch.no_grad()
    def score_frames(self, imgs: np.ndarray) -> float:
        """
        imgs: (N, H, W, 3) float32 [0,1]
        returns: float mean score 1.0-10.0
        """
        scores = [self.score_frame(img) for img in imgs]
        return float(np.mean(scores))

    def _parse_score(self, text: str) -> float:
        """生成テキストから数値を抽出。失敗時は 5.0 (中立)"""
        nums = re.findall(r'\b([1-9]|10)\b', text)
        if nums:
            return float(nums[0])
        # 数字が見つからない場合は中立スコア
        print(f"  [LLaVA] Parse failed: '{text}' → 5.0")
        return 5.0

    def vram_used_mb(self) -> int:
        if "cuda" in self.device:
            dev = torch.device(self.device)
            return torch.cuda.memory_allocated(dev) // (1024 ** 2)
        return 0

import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
from transformers import CLIPProcessor
import hpsv2
import hpsv2.src.open_clip as hps_open_clip
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


_BPE_VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"
_BPE_VOCAB_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def ensure_hpsv2_bpe_vocab():
    open_clip_dir = Path(hps_open_clip.__file__).resolve().parent
    vocab_path = open_clip_dir / _BPE_VOCAB_NAME
    if vocab_path.exists():
        return vocab_path

    open_clip_dir.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(_BPE_VOCAB_URL, vocab_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"HPSv2 tokenizer vocab is missing at {vocab_path}. "
            f"Download {_BPE_VOCAB_URL} to that path, or rerun with network access."
        ) from exc

    return vocab_path


class HPSv2Scorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision=self.dtype,
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
        ensure_hpsv2_bpe_vocab()
        # force download of model via score
        hpsv2.score([], "")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}["pixel_values"]
        text = self.tokenizer(prompts).to(self.device)
        outputs = self.model(inputs, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        scores = torch.diagonal(logits_per_image)

        return scores

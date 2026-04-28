from typing import Callable, Dict, Iterable, Optional

import torch
from PIL import Image


FINAL_REWARD_SCORERS = ("clip", "pick", "image_reward", "aesthetic", "hpsv2")

_SCORER_ALIASES = {
    "clip": "clip",
    "clip_score": "clip",
    "pick": "pick",
    "pick_score": "pick",
    "image_reward": "image_reward",
    "imagereward": "image_reward",
    "image_reward_score": "image_reward",
    "aesthetic": "aesthetic",
    "aesthetic_score": "aesthetic",
    "hpsv2": "hpsv2",
    "hps": "hpsv2",
    "hps_score": "hpsv2",
}


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    if images.min() < 0:
        images = ((images / 2) + 0.5).clamp(0, 1)
    return images


def normalize_reward_name(name: str) -> str:
    key = name.lower().strip()
    if key not in _SCORER_ALIASES:
        raise ValueError(f"Unsupported reward scorer: {name}")
    return _SCORER_ALIASES[key]


def _build_raw_scorer(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
):
    normalized = normalize_reward_name(name)

    if normalized == "clip":
        from seg.scorers.clip_scorer import CLIPScorer

        scorer = CLIPScorer(dtype=dtype, device=device)
        scorer.requires_grad_(False)
        return scorer, True

    if normalized == "pick":
        from seg.scorers.PickScore_scorer import PickScoreScorer

        scorer = PickScoreScorer(dtype=dtype, device=device)
        scorer.requires_grad_(False)
        return scorer, True

    if normalized == "image_reward":
        from seg.scorers.ImageReward_scorer import ImageRewardScorer

        scorer = ImageRewardScorer(dtype=dtype, device=device)
        scorer.requires_grad_(False)
        return scorer, True

    if normalized == "aesthetic":
        from seg.scorers.aesthetic_scorer import AestheticScorer

        scorer = AestheticScorer(dtype=dtype, device=device)
        scorer.requires_grad_(False)
        return scorer, False

    if normalized == "hpsv2":
        from seg.scorers.hpsv2_scorer import HPSv2Scorer

        scorer = HPSv2Scorer(dtype=dtype, device=device)
        scorer.requires_grad_(False)
        return scorer, True

    raise ValueError(f"Unsupported reward scorer: {name}")


def build_reward_scorer(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[torch.Tensor, Iterable[str]], torch.Tensor]:
    scorer, needs_prompts = _build_raw_scorer(name=name, dtype=dtype, device=device)

    def _score(images: torch.Tensor, prompts: Iterable[str]) -> torch.Tensor:
        normalized_images = _normalize_images(images)
        if needs_prompts:
            return scorer(normalized_images, prompts)
        return scorer(normalized_images)

    return _score


def build_final_reward_scorers(
    dtype: torch.dtype,
    device: torch.device,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Callable[[torch.Tensor, Iterable[str]], torch.Tensor]]:
    scorer_names = tuple(names) if names is not None else FINAL_REWARD_SCORERS
    return {normalize_reward_name(name): build_reward_scorer(name, dtype=dtype, device=device) for name in scorer_names}


def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np

    def loss_fn(images):
        images = _normalize_images(images)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images_pil = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images_pil]
        for image, buffer in zip(images_pil, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss
        return loss, rewards

    return loss_fn


def clip_score(inference_dtype=None, device=None, return_loss=False):
    scorer = build_reward_scorer("clip", dtype=torch.float32, device=device)

    if not return_loss:
        return scorer

    def loss_fn(images, prompts):
        scores = scorer(images, prompts)
        loss = -scores
        return loss, scores

    return loss_fn


def aesthetic_score(
    torch_dtype=None,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    scorer = build_reward_scorer("aesthetic", dtype=torch.float32, device=device)

    if not return_loss:
        return scorer

    def loss_fn(images, prompts):
        scores = scorer(images, prompts)
        if aesthetic_target is None:
            loss = -1 * scores
        else:
            loss = abs(scores - aesthetic_target)
        return loss * grad_scale, scores

    return loss_fn


def hps_score(inference_dtype=None, device=None, return_loss=False):
    scorer = build_reward_scorer("hpsv2", dtype=torch.float32, device=device)

    if not return_loss:
        return scorer

    def loss_fn(images, prompts):
        scores = scorer(images, prompts)
        loss = 1.0 - scores
        return loss, scores

    return loss_fn


def ImageReward(inference_dtype=None, device=None, return_loss=False):
    scorer = build_reward_scorer("image_reward", dtype=torch.float32, device=device)

    if not return_loss:
        return scorer

    def loss_fn(images, prompts):
        scores = scorer(images, prompts)
        loss = -scores
        return loss, scores

    return loss_fn


def PickScore(inference_dtype=None, device=None, return_loss=False):
    scorer = build_reward_scorer("pick", dtype=torch.float32, device=device)

    if not return_loss:
        return scorer

    def loss_fn(images, prompts):
        scores = scorer(images, prompts)
        loss = -scores
        return loss, scores

    return loss_fn

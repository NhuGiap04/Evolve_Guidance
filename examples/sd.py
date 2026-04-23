import argparse
import csv
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline

from config.sd import get_config
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer


def _to_lpips_input(images):
    if images.min() < 0:
        return images.clamp(-1, 1)
    return (images * 2 - 1).clamp(-1, 1)


def load_reference_images(ref_dir, expected_count, target_size_hw, device, dtype):
    ref_path = Path(ref_dir)
    if not ref_path.exists() or not ref_path.is_dir():
        raise ValueError(f"LPIPS reference directory not found: {ref_dir}")

    image_paths = [
        p for p in sorted(ref_path.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    if len(image_paths) < expected_count:
        raise ValueError(
            f"LPIPS requires at least {expected_count} reference images in {ref_dir}, found {len(image_paths)}."
        )

    target_h, target_w = target_size_hw
    loaded = []
    used_paths = image_paths[:expected_count]
    for image_path in used_paths:
        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB").resize((target_w, target_h), Image.BICUBIC)
        image_tensor = torch.from_numpy(np.asarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        loaded.append(image_tensor)

    refs = torch.stack(loaded, dim=0).to(device=device, dtype=dtype)
    return refs, used_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion v1.5 sampling with optional reward evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pick",
        choices=["pick", "clip", "seg"],
        help="Config preset name from config/sd.py.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A close up of a handpalm with leaves growing from it.",
        help="Prompt used for sampling.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt used during CFG guidance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/sd",
        help="Directory for generated images and CSV outputs.",
    )
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="image_reward",
        choices=["none", "clip", "pick", "image_reward", "lpips"],
        help="Optional second reward model for final image evaluation.",
    )
    parser.add_argument(
        "--lpips-ref-dir",
        type=str,
        default=None,
        help="Directory of reference images for LPIPS evaluation (sorted order, one per sample).",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g. cuda or cpu.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override.")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional num inference steps override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Optional CFG scale override.")
    parser.add_argument("--eta", type=float, default=None, help="Optional DDIM eta override.")

    return parser.parse_args()


def build_reward_scorer(name, dtype, device):
    normalized = name.lower()
    if normalized in {"pick", "pick_score"}:
        return PickScoreScorer(dtype=dtype, device=device)
    if normalized in {"clip", "clip_score"}:
        return CLIPScorer(dtype=dtype, device=device)
    if normalized in {"image_reward", "imagereward", "image_reward_score"}:
        return ImageRewardScorer(dtype=dtype, device=device)
    raise ValueError(f"Unsupported reward scorer: {name}")


def save_tensor_image(image_tensor, path):
    image_uint8 = (image_tensor.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    image_hwc = image_uint8.permute(1, 2, 0)
    Image.fromarray(image_hwc.numpy()).save(path)


def main():
    args = parse_args()

    config = get_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.num_steps is not None:
        config.sample.num_steps = args.num_steps
    if args.batch_size is not None:
        config.sample.batch_size = args.batch_size
    if args.guidance_scale is not None:
        config.sample.guidance_scale = args.guidance_scale
    if args.eta is not None:
        config.sample.eta = args.eta

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    inference_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=inference_dtype,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(config.sample.num_steps)
    pipe.safety_checker = None

    steer_scorer = build_reward_scorer(config.reward_fn, dtype=inference_dtype, device=device)
    eval_scorer = None
    if args.eval_reward not in {"none", "lpips"}:
        eval_scorer = build_reward_scorer(args.eval_reward, dtype=inference_dtype, device=device)

    out_dir = Path(args.output_dir) / f"{args.config}_seed{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [args.prompt] * config.sample.batch_size
    negative_prompts = [args.negative_prompt] * config.sample.batch_size
    generator = torch.Generator(device=device).manual_seed(config.seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            output_type="pt",
            generator=generator,
        )

    final_images = result.images if hasattr(result, "images") else result[0]

    with torch.no_grad():
        final_steer_scores = steer_scorer(
            final_images,
            [args.prompt] * final_images.shape[0],
        ).detach().float().cpu()

    for idx, image_tensor in enumerate(final_images):
        score = float(final_steer_scores[idx])
        file_name = f"sample_{idx:02d}_score_{score:.6f}.png"
        save_tensor_image(image_tensor, out_dir / file_name)

    if args.eval_reward == "lpips":
        if not args.lpips_ref_dir:
            raise ValueError("--lpips-ref-dir is required when --eval-reward lpips is selected.")

        with torch.no_grad():
            reference_images, reference_paths = load_reference_images(
                ref_dir=args.lpips_ref_dir,
                expected_count=final_images.shape[0],
                target_size_hw=(final_images.shape[-2], final_images.shape[-1]),
                device=device,
                dtype=inference_dtype,
            )
            lpips_metric = lpips.LPIPS(net=args.lpips_net).to(device)
            lpips_values = lpips_metric(
                _to_lpips_input(final_images).to(dtype=torch.float32),
                _to_lpips_input(reference_images).to(dtype=torch.float32),
            ).flatten().detach().float().cpu()

        lpips_csv_path = out_dir / "lpips_scores.csv"
        with lpips_csv_path.open("w", encoding="utf-8", newline="") as lpips_file:
            writer = csv.writer(lpips_file)
            writer.writerow(["sample_index", "reference_image", "lpips_distance"])
            for idx, score in enumerate(lpips_values.tolist()):
                writer.writerow([idx, reference_paths[idx].name, score])

        print(
            "Final LPIPS stats (lower is better): "
            f"mean={lpips_values.mean().item():.6f} "
            f"min={lpips_values.min().item():.6f} "
            f"max={lpips_values.max().item():.6f}"
        )
        print("Saved LPIPS scores to:", lpips_csv_path)

    elif eval_scorer is not None:
        with torch.no_grad():
            eval_prompts = [args.prompt] * final_images.shape[0]
            final_eval_scores = eval_scorer(final_images, eval_prompts).detach().float().cpu()
        print(
            f"Final eval reward stats ({args.eval_reward}): "
            f"mean={final_eval_scores.mean().item():.6f} max={final_eval_scores.max().item():.6f}"
        )

    print("Saved outputs to:", out_dir)
    print(
        "Final steering reward stats: "
        f"mean={final_steer_scores.mean().item():.6f} max={final_steer_scores.max().item():.6f}"
    )


if __name__ == "__main__":
    main()

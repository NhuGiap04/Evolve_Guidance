import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline

from config.sdxl import get_config
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed SDXL sampling with intermediate reward traces and plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pick",
        choices=["pick", "clip", "seg"],
        help="Config preset name from config/sdxl.py.",
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
        default="logs/examples/sdxl_detailed",
        help="Directory for generated images, traces, and plots.",
    )
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="image_reward",
        choices=["none", "clip", "pick", "image_reward"],
        help="Optional second reward model for tracing decoded steps.",
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


def decode_latents_sdxl(pipe, latents):
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        if torch.backends.mps.is_available():
            pipe.vae = pipe.vae.to(latents.dtype)
        else:
            latents = latents.to(pipe.vae.dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    do_denormalize = [True] * image.shape[0]
    return pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)


def save_tensor_image(image_tensor, path):
    image_np = (image_tensor.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(path)


def save_trace_plot(step_ids, mean_trace, max_trace, title, ylabel, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(step_ids, mean_trace, label="mean")
    plt.plot(step_ids, max_trace, label="max")
    plt.xlabel("Sampling step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
    load_kwargs = {"torch_dtype": inference_dtype, "use_safetensors": True}
    if inference_dtype == torch.float16:
        load_kwargs["variant"] = "fp16"

    pipe = DiffusionPipeline.from_pretrained(config.pretrained.model, **load_kwargs).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(config.sample.num_steps)

    # Keep VAE in fp32 for decode stability.
    # Keep text encoders in UNet/inference dtype to avoid cross-attention dtype mismatch.
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(dtype=inference_dtype)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(dtype=inference_dtype)

    steer_scorer = build_reward_scorer(config.reward_fn, dtype=inference_dtype, device=device)
    eval_scorer = None
    if args.eval_reward != "none":
        eval_scorer = build_reward_scorer(args.eval_reward, dtype=inference_dtype, device=device)

    out_dir = Path(args.output_dir) / f"{args.config}_seed{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [args.prompt] * config.sample.batch_size

    sample_size = pipe.unet.config.sample_size
    if isinstance(sample_size, int):
        latent_h, latent_w = sample_size, sample_size
    else:
        latent_h, latent_w = sample_size

    latents_0 = torch.randn(
        (
            config.sample.batch_size,
            pipe.unet.config.in_channels,
            latent_h,
            latent_w,
        ),
        device=device,
        dtype=inference_dtype,
    )

    all_latents = []
    trace_storage_dtype = torch.float16 if inference_dtype == torch.float16 else torch.float32

    def collect_step_latents(_pipe, _step_idx, _timestep, callback_kwargs):
        all_latents.append(callback_kwargs["latents"].detach().to("cpu", dtype=trace_storage_dtype))
        return callback_kwargs

    with torch.no_grad():
        result = pipe(
            prompt=prompts,
            negative_prompt=args.negative_prompt,
            num_inference_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            output_type="latent",
            latents=latents_0,
            callback_on_step_end=collect_step_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )

    final_latents = result.images if hasattr(result, "images") else result[0]
    final_images = decode_latents_sdxl(pipe, final_latents.to(device=device, dtype=inference_dtype))
    final_steer_scores = steer_scorer(final_images, prompts).detach().float().cpu().numpy()

    step_ids = np.arange(1, len(all_latents) + 1)
    steer_reward_mean_trace = []
    steer_reward_max_trace = []
    eval_reward_mean_trace = []
    eval_reward_max_trace = []

    for step_idx, step_latents_cpu in enumerate(all_latents, start=1):
        step_latents = step_latents_cpu.to(device=device, dtype=inference_dtype)
        step_images = decode_latents_sdxl(pipe, step_latents)

        steer_scores = steer_scorer(step_images, prompts).detach().float().cpu()
        steer_reward_mean_trace.append(steer_scores.mean().item())
        steer_reward_max_trace.append(steer_scores.max().item())

        log_msg = (
            f"[step {step_idx:03d}] "
            f"steer_mean={steer_scores.mean().item():.6f} "
            f"steer_max={steer_scores.max().item():.6f}"
        )

        if eval_scorer is not None:
            eval_scores = eval_scorer(step_images, prompts).detach().float().cpu()
            eval_reward_mean_trace.append(eval_scores.mean().item())
            eval_reward_max_trace.append(eval_scores.max().item())
            log_msg += (
                f" | eval_mean={eval_scores.mean().item():.6f}"
                f" eval_max={eval_scores.max().item():.6f}"
            )

        print(log_msg)

    np.save(out_dir / "steer_reward_mean_trace.npy", np.array(steer_reward_mean_trace, dtype=np.float32))
    np.save(out_dir / "steer_reward_max_trace.npy", np.array(steer_reward_max_trace, dtype=np.float32))

    save_trace_plot(
        step_ids,
        steer_reward_mean_trace,
        steer_reward_max_trace,
        title=f"Steering reward trace ({config.reward_fn})",
        ylabel="Reward",
        out_path=out_dir / "steer_reward_trace.png",
    )

    if eval_scorer is not None:
        np.save(out_dir / "eval_reward_mean_trace.npy", np.array(eval_reward_mean_trace, dtype=np.float32))
        np.save(out_dir / "eval_reward_max_trace.npy", np.array(eval_reward_max_trace, dtype=np.float32))
        save_trace_plot(
            step_ids,
            eval_reward_mean_trace,
            eval_reward_max_trace,
            title=f"Eval reward trace ({args.eval_reward})",
            ylabel="Reward",
            out_path=out_dir / "eval_reward_trace.png",
        )

    for idx, image_tensor in enumerate(final_images):
        score = float(final_steer_scores[idx])
        file_name = f"sample_{idx:02d}_steer_{score:.6f}.png"
        save_tensor_image(image_tensor, out_dir / file_name)

    print("Saved outputs to:", out_dir)
    print(
        "Final steering reward stats: "
        f"mean={final_steer_scores.mean():.6f} max={final_steer_scores.max():.6f}"
    )


if __name__ == "__main__":
    main()
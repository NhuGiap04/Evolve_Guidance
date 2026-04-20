import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline

from config.sdxl import get_config
from seg.diffusers_patch.pipeline_using_Stein_SDXL import pipeline_using_stein_sdxl
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed SDXL Stein-guided sampling with per-step reward traces and plots."
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
        default="logs/sdxl",
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

    parser.add_argument("--num-particles", type=int, default=None, help="Optional number of particles override.")
    parser.add_argument("--batch-p", type=int, default=None, help="Optional reward-gradient micro-batch particle count.")
    parser.add_argument("--stein-step", type=float, default=None, help="Optional Stein base step size override.")
    parser.add_argument("--stein-loop", type=int, default=None, help="Optional number of Stein inner loops override.")
    parser.add_argument("--stein-kernel", type=str, default=None, choices=["rbf"], help="Stein kernel.")
    parser.add_argument("--stein-adagrad-eps", type=float, default=None, help="Optional AdaGrad epsilon override.")
    parser.add_argument("--kl-coeff", type=float, default=None, help="Optional reward scaling denominator override.")
    parser.add_argument(
        "--steer-start",
        type=int,
        default=None,
        help="Steering start inference-step index (0-based, default: 0).",
    )
    parser.add_argument(
        "--steer-end",
        type=int,
        default=None,
        help="Steering end inference-step index (0-based, default: last step).",
    )

    parser.add_argument(
        "--save-intermediate-images",
        action="store_true",
        help="Decode and save intermediate decoded images for each denoising step.",
    )
    parser.add_argument(
        "--trace-decode-batch-size",
        type=int,
        default=1,
        help="How many latent samples to decode at once when tracing each denoising step.",
    )
    parser.add_argument(
        "--intermediate-max-samples",
        type=int,
        default=None,
        help="Optional cap on samples to save per step when --save-intermediate-images is used.",
    )

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


def save_before_after_plot(step_ids, pre_values, post_values, title, ylabel, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(step_ids, pre_values, label="before_steer", linewidth=2)
    plt.plot(step_ids, post_values, label="after_steer", linewidth=2)
    plt.xlabel("Steered step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def release_generation_modules(pipe):
    # Per-step trace decoding only needs the VAE.
    pipe.unet.to("cpu")
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to("cpu")
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to("cpu")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.trace_decode_batch_size < 1:
        args.trace_decode_batch_size = 1

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

    if args.num_particles is not None:
        config.sample.num_particles = args.num_particles
    if args.batch_p is not None:
        config.sample.batch_p = args.batch_p
    if args.stein_step is not None:
        config.sample.stein_step = args.stein_step
    if args.stein_loop is not None:
        config.sample.stein_loop = args.stein_loop
    if args.stein_kernel is not None:
        config.sample.stein_kernel = args.stein_kernel
    if args.stein_adagrad_eps is not None:
        config.sample.stein_adagrad_eps = args.stein_adagrad_eps
    if args.kl_coeff is not None:
        config.sample.kl_coeff = args.kl_coeff
    if args.steer_start is not None:
        config.sample.steer_start = args.steer_start
    if args.steer_end is not None:
        config.sample.steer_end = args.steer_end

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
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")

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
    intermediate_out_dir = out_dir / "intermediate_images"
    if args.save_intermediate_images:
        intermediate_out_dir.mkdir(parents=True, exist_ok=True)

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

    call_kwargs = dict(
        prompt=prompts,
        negative_prompt=args.negative_prompt,
        num_inference_steps=config.sample.num_steps,
        guidance_scale=config.sample.guidance_scale,
        eta=config.sample.eta,
        output_type="latent",
        latents=latents_0,
        reward_fn=steer_scorer,
        num_particles=config.sample.num_particles,
        batch_p=config.sample.batch_p,
        stein_step=config.sample.stein_step,
        stein_loop=config.sample.stein_loop,
        stein_kernel=config.sample.stein_kernel,
        stein_adagrad_eps=config.sample.stein_adagrad_eps,
        stein_adagrad_clip=config.sample.stein_adagrad_clip,
        kl_coeff=config.sample.kl_coeff,
        steer_start=config.sample.steer_start,
        steer_end=config.sample.steer_end,
        intermediate_rewards=config.sample.intermediate_rewards,
        return_all_particles=True,
        return_dict=False,
    )

    if args.save_intermediate_images:
        call_kwargs["callback_on_step_end"] = collect_step_latents
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    with torch.no_grad():
        # Call the local Stein implementation explicitly so traces/logs are produced.
        result = pipeline_using_stein_sdxl(pipe, **call_kwargs)

    final_latents = result[0] if isinstance(result, (tuple, list)) else result
    intermediate_logs = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else {}

    with torch.no_grad():
        final_images = decode_latents_sdxl(pipe, final_latents.to(device=device, dtype=inference_dtype))
        final_steer_scores = steer_scorer(
            final_images,
            [args.prompt] * final_images.shape[0],
        ).detach().float().cpu().numpy()

    step_indices = np.array(intermediate_logs.get("step_indices", []), dtype=np.int32)
    timesteps = np.array(intermediate_logs.get("timesteps", []), dtype=np.int32)
    pre_mean = np.array(intermediate_logs.get("pre_steer_mean", []), dtype=np.float32)
    post_mean = np.array(intermediate_logs.get("post_steer_mean", []), dtype=np.float32)
    pre_max = np.array(intermediate_logs.get("pre_steer_max", []), dtype=np.float32)
    post_max = np.array(intermediate_logs.get("post_steer_max", []), dtype=np.float32)

    for idx in range(len(pre_mean)):
        print(
            f"[steer step {int(step_indices[idx]):03d} | t={int(timesteps[idx]):04d}] "
            f"mean: {pre_mean[idx]:.6f} -> {post_mean[idx]:.6f} "
            f"(delta={post_mean[idx] - pre_mean[idx]:+.6f}) | "
            f"max: {pre_max[idx]:.6f} -> {post_max[idx]:.6f} "
            f"(delta={post_max[idx] - pre_max[idx]:+.6f})"
        )

    np.save(out_dir / "steer_step_indices.npy", step_indices)
    np.save(out_dir / "steer_timesteps.npy", timesteps)
    np.save(out_dir / "steer_pre_mean.npy", pre_mean)
    np.save(out_dir / "steer_post_mean.npy", post_mean)
    np.save(out_dir / "steer_pre_max.npy", pre_max)
    np.save(out_dir / "steer_post_max.npy", post_max)

    plot_x = np.arange(1, len(pre_mean) + 1)
    save_before_after_plot(
        plot_x,
        pre_mean,
        post_mean,
        title=f"Before/After steering reward ({config.reward_fn}) - mean",
        ylabel="Reward",
        out_path=out_dir / "steer_before_after_mean.png",
    )
    save_before_after_plot(
        plot_x,
        pre_max,
        post_max,
        title=f"Before/After steering reward ({config.reward_fn}) - max",
        ylabel="Reward",
        out_path=out_dir / "steer_before_after_max.png",
    )

    if args.save_intermediate_images and len(all_latents) > 0:
        for step_idx, step_latents_cpu in enumerate(all_latents, start=1):
            for offset in range(0, step_latents_cpu.shape[0], args.trace_decode_batch_size):
                step_chunk = step_latents_cpu[offset : offset + args.trace_decode_batch_size]

                with torch.inference_mode():
                    step_latents = step_chunk.to(device=device, dtype=inference_dtype)
                    step_images = decode_latents_sdxl(pipe, step_latents)
                    chunk_prompts = [args.prompt] * step_images.shape[0]
                    steer_scores = steer_scorer(step_images, chunk_prompts).detach().float().cpu()

                    max_samples = step_images.shape[0]
                    if args.intermediate_max_samples is not None:
                        max_samples = min(max_samples, max(args.intermediate_max_samples, 0))

                    for local_idx in range(max_samples):
                        sample_idx = offset + local_idx
                        steer_score = float(steer_scores[local_idx].item())
                        file_name = (
                            f"step_{step_idx:03d}_sample_{sample_idx:03d}"
                            f"_steer_{steer_score:.6f}.png"
                        )
                        save_tensor_image(step_images[local_idx], intermediate_out_dir / file_name)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if device.type == "cuda":
        release_generation_modules(pipe)

    for idx, image_tensor in enumerate(final_images):
        score = float(final_steer_scores[idx])
        file_name = f"sample_{idx:02d}_steer_{score:.6f}.png"
        save_tensor_image(image_tensor, out_dir / file_name)

    if eval_scorer is not None:
        with torch.no_grad():
            eval_prompts = [args.prompt] * final_images.shape[0]
            final_eval_scores = eval_scorer(final_images, eval_prompts).detach().float().cpu().numpy()
        print(
            f"Final eval reward stats ({args.eval_reward}): "
            f"mean={final_eval_scores.mean():.6f} max={final_eval_scores.max():.6f}"
        )

    print("Saved outputs to:", out_dir)
    print(
        "Final steering reward stats: "
        f"mean={final_steer_scores.mean():.6f} max={final_steer_scores.max():.6f}"
    )


if __name__ == "__main__":
    main()

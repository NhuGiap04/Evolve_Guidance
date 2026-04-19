import argparse
import json
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
        description="SDXL Stein-guided sampling with optional intermediate steering logs."
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
        default="logs/examples/sdxl_stein",
        help="Directory for generated images and logs.",
    )
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="none",
        choices=["none", "clip", "pick", "image_reward"],
        help="Optional second reward model for final-image evaluation.",
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
        "--save-logs",
        dest="save_logs",
        action="store_true",
        help="Save intermediate steering rewards and plots.",
    )
    parser.add_argument(
        "--no-save-logs",
        dest="save_logs",
        action="store_false",
        help="Skip intermediate steering logs/plots to save time.",
    )
    parser.set_defaults(save_logs=True)

    parser.add_argument(
        "--save-intermediate-images",
        action="store_true",
        help="When save_logs is enabled, also decode and save before/after steered intermediate images.",
    )
    parser.add_argument(
        "--show-intermediate-rewards",
        action="store_true",
        help="Print per-step pre/post steering rewards during sampling.",
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


def save_before_after_plot(step_ids, pre_values, post_values, title, out_path, ylabel="Reward"):
    plt.figure(figsize=(10, 5))
    plt.plot(step_ids, pre_values, label="before_steer", linewidth=2)
    plt.plot(step_ids, post_values, label="after_steer", linewidth=2)
    plt.xlabel("Steered timestep index")
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

    config.sample.show_intermediate_rewards = bool(args.show_intermediate_rewards)

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
    pipe.__call__ = pipeline_using_stein_sdxl.__get__(pipe, pipe.__class__)
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

    intermediate_pairs = []

    def collect_before_after_latents(_pipe, step_idx, timestep, callback_kwargs):
        pre = callback_kwargs.get("pre_stein_latents", None)
        post = callback_kwargs.get("post_stein_latents", None)
        if pre is not None and post is not None and args.save_intermediate_images:
            # Save only the first latent for visualization to control storage.
            intermediate_pairs.append(
                {
                    "step_idx": int(step_idx),
                    "timestep": int(timestep.item() if torch.is_tensor(timestep) else timestep),
                    "pre": pre[0:1].detach().cpu(),
                    "post": post[0:1].detach().cpu(),
                }
            )
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
        show_intermediate_rewards=config.sample.show_intermediate_rewards,
        return_dict=False,
        return_intermediate_rewards=args.save_logs,
    )

    if args.save_intermediate_images and args.save_logs:
        call_kwargs["callback_on_step_end"] = collect_before_after_latents
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents", "pre_stein_latents", "post_stein_latents"]

    with torch.no_grad():
        result = pipe(**call_kwargs)

    intermediate_logs = None
    if isinstance(result, dict):
        final_latents = result.get("images", None)
        if final_latents is None:
            raise ValueError("Pipeline dict output does not contain 'images'.")
        if args.save_logs:
            intermediate_logs = result.get("intermediate_rewards", None)
    elif isinstance(result, (tuple, list)):
        if len(result) == 0:
            raise ValueError("Pipeline returned an empty tuple/list.")
        final_latents = result[0]
        if args.save_logs and len(result) > 1:
            intermediate_logs = result[1]
    else:
        final_latents = result.images if hasattr(result, "images") else result

    if args.save_logs and intermediate_logs is None:
        print(
            "Warning: save_logs=True but intermediate logs were not returned by the pipeline. "
            "Continuing with final outputs only."
        )

    with torch.no_grad():
        final_images = decode_latents_sdxl(pipe, final_latents.to(device=device, dtype=inference_dtype))
        final_steer_scores = steer_scorer(final_images, prompts).detach().float().cpu().numpy()

    final_eval_scores = None
    if eval_scorer is not None:
        with torch.no_grad():
            final_eval_scores = eval_scorer(final_images, prompts).detach().float().cpu().numpy()

    for idx, image_tensor in enumerate(final_images):
        score = float(final_steer_scores[idx])
        file_name = f"sample_{idx:02d}_steer_{score:.6f}.png"
        save_tensor_image(image_tensor, out_dir / file_name)

    summary = {
        "config": args.config,
        "seed": int(config.seed),
        "num_steps": int(config.sample.num_steps),
        "num_particles": int(config.sample.num_particles),
        "stein_loop": int(config.sample.stein_loop),
        "stein_step": float(config.sample.stein_step),
        "steer_start": config.sample.steer_start,
        "steer_end": config.sample.steer_end,
        "save_logs": bool(args.save_logs),
        "final_steer_reward_mean": float(final_steer_scores.mean()),
        "final_steer_reward_max": float(final_steer_scores.max()),
        "final_steer_reward_values": [float(v) for v in final_steer_scores.tolist()],
    }

    if final_eval_scores is not None:
        summary["final_eval_reward_name"] = args.eval_reward
        summary["final_eval_reward_mean"] = float(final_eval_scores.mean())
        summary["final_eval_reward_max"] = float(final_eval_scores.max())
        summary["final_eval_reward_values"] = [float(v) for v in final_eval_scores.tolist()]

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_logs and intermediate_logs is not None:
        timesteps = np.array(intermediate_logs.get("timesteps", []), dtype=np.int32)
        pre_mean = np.array(intermediate_logs.get("pre_steer_mean", []), dtype=np.float32)
        post_mean = np.array(intermediate_logs.get("post_steer_mean", []), dtype=np.float32)
        pre_max = np.array(intermediate_logs.get("pre_steer_max", []), dtype=np.float32)
        post_max = np.array(intermediate_logs.get("post_steer_max", []), dtype=np.float32)

        np.save(out_dir / "steer_timesteps.npy", timesteps)
        np.save(out_dir / "steer_pre_mean.npy", pre_mean)
        np.save(out_dir / "steer_post_mean.npy", post_mean)
        np.save(out_dir / "steer_pre_max.npy", pre_max)
        np.save(out_dir / "steer_post_max.npy", post_max)

        save_before_after_plot(
            np.arange(1, len(pre_mean) + 1),
            pre_mean,
            post_mean,
            title=f"Intermediate reward before/after steering ({config.reward_fn}) - mean",
            out_path=out_dir / "steer_before_after_mean.png",
            ylabel="Reward",
        )
        save_before_after_plot(
            np.arange(1, len(pre_max) + 1),
            pre_max,
            post_max,
            title=f"Intermediate reward before/after steering ({config.reward_fn}) - max",
            out_path=out_dir / "steer_before_after_max.png",
            ylabel="Reward",
        )

        with open(out_dir / "intermediate_rewards.json", "w") as f:
            json.dump(intermediate_logs, f, indent=2)

        if args.save_intermediate_images and len(intermediate_pairs) > 0:
            inter_dir = out_dir / "intermediate_images"
            inter_dir.mkdir(parents=True, exist_ok=True)

            for item in intermediate_pairs:
                pre_latent = item["pre"].to(device=device, dtype=inference_dtype)
                post_latent = item["post"].to(device=device, dtype=inference_dtype)
                with torch.no_grad():
                    pre_img = decode_latents_sdxl(pipe, pre_latent)[0]
                    post_img = decode_latents_sdxl(pipe, post_latent)[0]
                    pre_score = float(steer_scorer(pre_img.unsqueeze(0), [args.prompt])[0].detach().float().cpu().item())
                    post_score = float(steer_scorer(post_img.unsqueeze(0), [args.prompt])[0].detach().float().cpu().item())

                step_idx = item["step_idx"]
                timestep = item["timestep"]
                save_tensor_image(
                    pre_img,
                    inter_dir / f"step_{step_idx:03d}_t{timestep:04d}_before_{pre_score:.6f}.png",
                )
                save_tensor_image(
                    post_img,
                    inter_dir / f"step_{step_idx:03d}_t{timestep:04d}_after_{post_score:.6f}.png",
                )

    print("Saved outputs to:", out_dir)
    print(
        "Final steering reward stats: "
        f"mean={final_steer_scores.mean():.6f} max={final_steer_scores.max():.6f}"
    )


if __name__ == "__main__":
    main()

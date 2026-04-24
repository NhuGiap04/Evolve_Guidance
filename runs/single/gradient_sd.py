import argparse
import csv
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline

from config.sd import get_config
from seg.diffusers_patch.pipeline_using_gradient_SD import pipeline_using_gradient_sd
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed SD Stein-guided sampling with per-step reward traces and plots."
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
        help="Directory for generated images, traces, and plots.",
    )
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="image_reward",
        choices=["none", "clip", "pick", "image_reward"],
        help="Optional second reward model used when --run-eval-now is enabled.",
    )
    parser.add_argument(
        "--run-eval-now",
        action="store_true",
        help="Run final/trace reward evaluation immediately. By default, evaluation is deferred.",
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
        help="Decode and save deferred intermediate images from saved trace latents.",
    )
    parser.add_argument(
        "--save-intermediate-rewards",
        action="store_true",
        help="Evaluate and save deferred intermediate steer/eval rewards (CSV + plots).",
    )
    parser.add_argument(
        "--trace-decode-batch-size",
        type=int,
        default=1,
        help="How many latent samples to decode at once when saving trace images.",
    )
    parser.add_argument(
        "--trace-eval-batch",
        type=int,
        default=1,
        help="How many latent samples to decode/score at once for deferred trace reward evaluation.",
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


def decode_latents_sd(pipe, latents):
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
    image_uint8 = (image_tensor.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    image_hwc = image_uint8.permute(1, 2, 0)
    Image.fromarray(image_hwc.numpy()).save(path)


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


def _expand_prompts_for_particles(prompts, num_particles):
    expanded = []
    for prompt in prompts:
        expanded.extend([prompt] * num_particles)
    return expanded


def _score_latents_in_batches(
    pipe,
    latents_cpu,
    prompts,
    steer_scorer,
    eval_scorer,
    batch_size,
    device,
    inference_dtype,
):
    steer_chunks = []
    eval_chunks = [] if eval_scorer is not None else None

    for offset in range(0, latents_cpu.shape[0], batch_size):
        chunk_cpu = latents_cpu[offset : offset + batch_size]

        with torch.inference_mode():
            chunk_latents = chunk_cpu.to(device=device, dtype=inference_dtype)
            chunk_images = decode_latents_sd(pipe, chunk_latents)
            chunk_prompts = prompts[offset : offset + chunk_images.shape[0]]

            chunk_steer = steer_scorer(chunk_images, chunk_prompts).detach().float().cpu()
            steer_chunks.append(chunk_steer)

            if eval_scorer is not None:
                chunk_eval = eval_scorer(chunk_images, chunk_prompts).detach().float().cpu()
                eval_chunks.append(chunk_eval)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    steer_scores = torch.cat(steer_chunks, dim=0) if steer_chunks else torch.empty(0, dtype=torch.float32)
    eval_scores = None
    if eval_chunks is not None:
        eval_scores = torch.cat(eval_chunks, dim=0) if eval_chunks else torch.empty(0, dtype=torch.float32)

    return steer_scores, eval_scores


def _save_intermediate_step_images(
    step_latents_cpu_list,
    intermediate_out_dir,
    pipe,
    steer_scorer,
    prompt,
    device,
    inference_dtype,
    decode_batch_size,
    intermediate_max_samples,
):
    if decode_batch_size < 1:
        decode_batch_size = 1

    sample_cap = None
    if intermediate_max_samples is not None:
        sample_cap = max(intermediate_max_samples, 0)

    for step_idx, step_latents_cpu in enumerate(step_latents_cpu_list, start=1):
        if sample_cap == 0:
            continue

        limit = step_latents_cpu.shape[0] if sample_cap is None else min(step_latents_cpu.shape[0], sample_cap)

        for offset in range(0, limit, decode_batch_size):
            chunk = step_latents_cpu[offset : min(limit, offset + decode_batch_size)]
            with torch.inference_mode():
                chunk_latents = chunk.to(device=device, dtype=inference_dtype)
                chunk_images = decode_latents_sd(pipe, chunk_latents)
                chunk_prompts = [prompt] * chunk_images.shape[0]
                chunk_scores = steer_scorer(chunk_images, chunk_prompts).detach().float().cpu()
                for local_idx, image in enumerate(chunk_images):
                    sample_idx = offset + local_idx
                    steer_score = float(chunk_scores[local_idx].item())
                    file_name = (
                        f"step_{step_idx:03d}_sample_{sample_idx:03d}"
                        f"_steer_{steer_score:.6f}.png"
                    )
                    save_tensor_image(image, intermediate_out_dir / file_name)

            if device.type == "cuda":
                torch.cuda.empty_cache()


def release_generation_modules(pipe):
    # Deferred trace decoding/scoring only needs the VAE.
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
    if args.trace_eval_batch < 1:
        args.trace_eval_batch = 1

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
    load_kwargs = {"torch_dtype": inference_dtype}
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        **load_kwargs,
    ).to(device)
    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(config.sample.num_steps)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")

    # Keep VAE in fp32 for decode stability.
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(dtype=inference_dtype)

    steer_scorer = build_reward_scorer(config.reward_fn, dtype=inference_dtype, device=device)
    eval_scorer = None
    if args.run_eval_now and args.eval_reward != "none":
        eval_scorer = build_reward_scorer(args.eval_reward, dtype=inference_dtype, device=device)

    out_dir = Path(args.output_dir) / f"{args.config}_seed{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_out_dir = out_dir / "intermediate_images"
    if args.save_intermediate_images:
        intermediate_out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [args.prompt] * config.sample.batch_size
    prompt_particles = _expand_prompts_for_particles(prompts, config.sample.num_particles)

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

    trace_entries = []
    step_latents_for_images = []
    trace_storage_dtype = torch.float16 if inference_dtype == torch.float16 else torch.float32

    def collect_step_latents(_pipe, step_idx, timestep, callback_kwargs):
        if args.save_intermediate_images:
            latents_at_step = callback_kwargs.get("latents")
            if latents_at_step is not None:
                step_latents_for_images.append(latents_at_step.detach().to("cpu", dtype=trace_storage_dtype))

        if args.save_intermediate_rewards:
            pre_x0 = callback_kwargs.get("pre_stein_pred_x0")
            post_x0 = callback_kwargs.get("post_stein_pred_x0")
            if pre_x0 is not None and post_x0 is not None:
                t_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
                trace_entries.append(
                    {
                        "step_index": int(step_idx),
                        "timestep": t_value,
                        "pre_x0_latents_cpu": pre_x0.detach().to("cpu", dtype=trace_storage_dtype),
                        "post_x0_latents_cpu": post_x0.detach().to("cpu", dtype=trace_storage_dtype),
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
        intermediate_rewards=False,
        return_all_particles=True,
        return_dict=False,
    )
    if args.save_intermediate_images or args.save_intermediate_rewards:
        call_kwargs["callback_on_step_end"] = collect_step_latents
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    inference_start = time.time()
    with torch.no_grad():
        result = pipeline_using_gradient_sd(pipe, **call_kwargs)
    inference_elapsed = time.time() - inference_start

    final_latents = result[0] if isinstance(result, (tuple, list)) else result

    with torch.no_grad():
        final_images = decode_latents_sd(pipe, final_latents.to(device=device, dtype=inference_dtype))

    # 1) Save final particle images first.
    for idx, image_tensor in enumerate(final_images):
        file_name = f"sample_{idx:02d}.png"
        save_tensor_image(image_tensor, out_dir / file_name)

    if device.type == "cuda":
        release_generation_modules(pipe)

    final_prompts = prompt_particles[: final_images.shape[0]]
    if len(final_prompts) != final_images.shape[0]:
        final_prompts = [args.prompt] * final_images.shape[0]

    # 2) Persist generation outputs now and optionally evaluate in-process.
    final_steer_scores = None
    final_eval_scores = None
    if args.run_eval_now:
        with torch.no_grad():
            final_steer_scores = steer_scorer(final_images, final_prompts).detach().float().cpu()
            if eval_scorer is not None:
                final_eval_scores = eval_scorer(final_images, final_prompts).detach().float().cpu()

    image_files = [f"sample_{idx:02d}.png" for idx in range(final_images.shape[0])]

    final_rewards_payload = {
        "prompt": args.prompt,
        "config": args.config,
        "num_images": int(final_images.shape[0]),
        "inference_time_sec": float(inference_elapsed),
        "image_files": image_files,
        "evaluation_deferred": not args.run_eval_now,
        "steer_reward_name": config.reward_fn,
        "steer_rewards": [float(v) for v in final_steer_scores.tolist()] if final_steer_scores is not None else None,
        "steer_reward_stats": (
            {
                "mean": float(final_steer_scores.mean().item()),
                "max": float(final_steer_scores.max().item()),
                "min": float(final_steer_scores.min().item()),
            }
            if final_steer_scores is not None
            else None
        ),
    }

    if final_eval_scores is not None:
        final_rewards_payload["eval_reward_name"] = args.eval_reward
        final_rewards_payload["eval_rewards"] = [float(v) for v in final_eval_scores.tolist()]
        final_rewards_payload["eval_reward_stats"] = {
            "mean": float(final_eval_scores.mean().item()),
            "max": float(final_eval_scores.max().item()),
            "min": float(final_eval_scores.min().item()),
        }
    else:
        final_rewards_payload["eval_reward_name"] = "none"
        final_rewards_payload["eval_rewards"] = None
        final_rewards_payload["eval_reward_stats"] = None

    final_rewards_path = out_dir / "final_rewards.json"
    with final_rewards_path.open("w", encoding="utf-8") as f:
        json.dump(final_rewards_payload, f, indent=2)

    if args.save_intermediate_rewards and not args.run_eval_now and len(trace_entries) > 0:
        deferred_trace_path = out_dir / "steer_trace_latents.pt"
        torch.save(trace_entries, deferred_trace_path)
        print(f"Saved deferred trace latents to: {deferred_trace_path}")

    if args.save_intermediate_rewards and args.run_eval_now:
        # 3) Deferred intermediate reward evaluation in trace-eval micro-batches.
        trace_rows = []
        pre_steer_mean = []
        post_steer_mean = []
        pre_steer_max = []
        post_steer_max = []
        pre_eval_mean = []
        post_eval_mean = []
        pre_eval_max = []
        post_eval_max = []

        for trace in trace_entries:
            trace_prompts = prompt_particles[: trace["pre_x0_latents_cpu"].shape[0]]
            if len(trace_prompts) != trace["pre_x0_latents_cpu"].shape[0]:
                trace_prompts = [args.prompt] * trace["pre_x0_latents_cpu"].shape[0]

            pre_steer_scores, pre_eval_scores = _score_latents_in_batches(
                pipe=pipe,
                latents_cpu=trace["pre_x0_latents_cpu"],
                prompts=trace_prompts,
                steer_scorer=steer_scorer,
                eval_scorer=eval_scorer,
                batch_size=args.trace_eval_batch,
                device=device,
                inference_dtype=inference_dtype,
            )
            post_steer_scores, post_eval_scores = _score_latents_in_batches(
                pipe=pipe,
                latents_cpu=trace["post_x0_latents_cpu"],
                prompts=trace_prompts,
                steer_scorer=steer_scorer,
                eval_scorer=eval_scorer,
                batch_size=args.trace_eval_batch,
                device=device,
                inference_dtype=inference_dtype,
            )

            row = {
                "step_index": int(trace["step_index"]),
                "timestep": int(trace["timestep"]),
                "pre_steer_mean": float(pre_steer_scores.mean().item()),
                "post_steer_mean": float(post_steer_scores.mean().item()),
                "pre_steer_max": float(pre_steer_scores.max().item()),
                "post_steer_max": float(post_steer_scores.max().item()),
            }

            if eval_scorer is not None and pre_eval_scores is not None and post_eval_scores is not None:
                row["pre_eval_mean"] = float(pre_eval_scores.mean().item())
                row["post_eval_mean"] = float(post_eval_scores.mean().item())
                row["pre_eval_max"] = float(pre_eval_scores.max().item())
                row["post_eval_max"] = float(post_eval_scores.max().item())

            trace_rows.append(row)

            pre_steer_mean.append(row["pre_steer_mean"])
            post_steer_mean.append(row["post_steer_mean"])
            pre_steer_max.append(row["pre_steer_max"])
            post_steer_max.append(row["post_steer_max"])

            print(
                f"[steer step {row['step_index']:03d} | t={row['timestep']:04d}] "
                f"mean: {row['pre_steer_mean']:.6f} -> {row['post_steer_mean']:.6f} "
                f"(delta={row['post_steer_mean'] - row['pre_steer_mean']:+.6f}) | "
                f"max: {row['pre_steer_max']:.6f} -> {row['post_steer_max']:.6f} "
                f"(delta={row['post_steer_max'] - row['pre_steer_max']:+.6f})"
            )

            if "pre_eval_mean" in row:
                pre_eval_mean.append(row["pre_eval_mean"])
                post_eval_mean.append(row["post_eval_mean"])
                pre_eval_max.append(row["pre_eval_max"])
                post_eval_max.append(row["post_eval_max"])
                print(
                    f"[eval  step {row['step_index']:03d} | t={row['timestep']:04d}] "
                    f"mean: {row['pre_eval_mean']:.6f} -> {row['post_eval_mean']:.6f} "
                    f"(delta={row['post_eval_mean'] - row['pre_eval_mean']:+.6f}) | "
                    f"max: {row['pre_eval_max']:.6f} -> {row['post_eval_max']:.6f} "
                    f"(delta={row['post_eval_max'] - row['pre_eval_max']:+.6f})"
                )

        # 4) Save combined trace CSV and before/after plots.
        trace_csv_path = out_dir / "steer_trace.csv"
        with trace_csv_path.open("w", encoding="utf-8", newline="") as trace_file:
            fieldnames = [
                "step_index",
                "timestep",
                "pre_steer_mean",
                "post_steer_mean",
                "pre_steer_max",
                "post_steer_max",
            ]
            if eval_scorer is not None:
                fieldnames.extend(["pre_eval_mean", "post_eval_mean", "pre_eval_max", "post_eval_max"])

            writer = csv.DictWriter(trace_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in trace_rows:
                writer.writerow(row)

        if len(trace_rows) > 0:
            plot_x = list(range(1, len(trace_rows) + 1))
            save_before_after_plot(
                plot_x,
                pre_steer_mean,
                post_steer_mean,
                title=f"Before/After steering reward ({config.reward_fn}) - mean",
                ylabel="Reward",
                out_path=out_dir / "steer_before_after_mean.png",
            )
            save_before_after_plot(
                plot_x,
                pre_steer_max,
                post_steer_max,
                title=f"Before/After steering reward ({config.reward_fn}) - max",
                ylabel="Reward",
                out_path=out_dir / "steer_before_after_max.png",
            )

            if eval_scorer is not None and len(pre_eval_mean) == len(trace_rows):
                save_before_after_plot(
                    plot_x,
                    pre_eval_mean,
                    post_eval_mean,
                    title=f"Before/After eval reward ({args.eval_reward}) - mean",
                    ylabel="Reward",
                    out_path=out_dir / "eval_before_after_mean.png",
                )
                save_before_after_plot(
                    plot_x,
                    pre_eval_max,
                    post_eval_max,
                    title=f"Before/After eval reward ({args.eval_reward}) - max",
                    ylabel="Reward",
                    out_path=out_dir / "eval_before_after_max.png",
                )

    if args.save_intermediate_images and len(step_latents_for_images) > 0:
        _save_intermediate_step_images(
            step_latents_cpu_list=step_latents_for_images,
            intermediate_out_dir=intermediate_out_dir,
            pipe=pipe,
            steer_scorer=steer_scorer,
            prompt=args.prompt,
            device=device,
            inference_dtype=inference_dtype,
            decode_batch_size=args.trace_decode_batch_size,
            intermediate_max_samples=args.intermediate_max_samples,
        )

    if final_eval_scores is not None:
        print(
            f"Final eval reward stats ({args.eval_reward}): "
            f"mean={final_eval_scores.mean().item():.6f} max={final_eval_scores.max().item():.6f}"
        )

    print("Saved outputs to:", out_dir)
    print(f"Inference time (pipeline only): {inference_elapsed:.4f}s")
    if final_steer_scores is not None:
        print(
            "Final steering reward stats: "
            f"mean={final_steer_scores.mean().item():.6f} max={final_steer_scores.max().item():.6f}"
        )
    else:
        print("Post-generation evaluation deferred. Run a separate evaluator on saved outputs.")


if __name__ == "__main__":
    main()

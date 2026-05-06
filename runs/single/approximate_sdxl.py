import argparse
import csv
import gc
import json
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline

from config.sdxl import get_config
from seg.diffusers_patch.pipeline_using_approximate_SDXL import pipeline_using_approximate_sdxl
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate SDXL Stein-guided sampling with per-step reward traces and plots."
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
        default="logs/sdxl_approx",
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
    parser.add_argument(
        "--offload",
        type=str,
        default="none",
        choices=["none", "model", "sequential"],
        help="Enable CPU offload to reduce VRAM (requires accelerate).",
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
        "--reward-guidance-rho",
        type=float,
        default=None,
        help="Optional reward guidance coefficient multiplier override.",
    )
    parser.add_argument(
        "--reward-scale-fixed",
        type=float,
        default=None,
        help="Optional fixed reward scaling coefficient override.",
    )
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
        "--x0-anchor-model",
        type=str,
        default=None,
        choices=["base", "dpm", "lcm", "dmd2"],
        help="Anchor predictor for x0 during reward evaluation (base, dpm, lcm, dmd2).",
    )
    parser.add_argument(
        "--x0-anchor-steps",
        type=int,
        default=None,
        help="Number of anchor solver steps when using dpm/lcm (>=1).",
    )
    parser.add_argument(
        "--x0-anchor-lora-path",
        type=str,
        default=None,
        help="Optional LoRA path for LCM anchor prediction.",
    )
    parser.add_argument(
        "--x0-anchor-lora-scale",
        type=float,
        default=None,
        help="Optional LoRA scale for LCM anchor prediction.",
    )
    parser.add_argument(
        "--detach-anchors",
        action="store_true",
        help="Detach x0 anchors so reward does not backprop through anchor prediction.",
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
        "--show-intermediate-rewards",
        action="store_true",
        help="Print per-step pre/post steering rewards during inference without saving trace files.",
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


def _download_lora_if_missing(url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading LoRA to {target_path}")
    with urllib.request.urlopen(url) as response, open(target_path, "wb") as handle:
        handle.write(response.read())


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
            chunk_images = decode_latents_sdxl(pipe, chunk_latents)
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
    step_count = len(step_latents_cpu_list)
    for step_idx, latents_cpu in enumerate(step_latents_cpu_list):
        if intermediate_max_samples is not None:
            latents_cpu = latents_cpu[:intermediate_max_samples]

        print(f"[INFO] Decoding step {step_idx + 1}/{step_count} ({latents_cpu.shape[0]} samples)")
        with torch.no_grad():
            step_scores, _ = _score_latents_in_batches(
                pipe,
                latents_cpu,
                [prompt] * latents_cpu.shape[0],
                steer_scorer,
                None,
                decode_batch_size,
                device,
                inference_dtype,
            )

            images = decode_latents_sdxl(pipe, latents_cpu.to(device=device, dtype=inference_dtype))

        step_dir = intermediate_out_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images.shape[0]):
            img_path = step_dir / f"sample_{i:02d}.png"
            save_tensor_image(images[i], img_path)

        scores_path = step_dir / "reward_scores.csv"
        with scores_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "reward"])
            for i, score in enumerate(step_scores.tolist()):
                writer.writerow([i, f"{score:.6f}"])

        if device.type == "cuda":
            torch.cuda.empty_cache()


def _save_intermediate_step_rewards(
    trace_entries,
    out_dir,
    prompt,
    steer_scorer,
    eval_scorer,
    pipe,
    device,
    inference_dtype,
    batch_size,
):
    if not trace_entries:
        return

    trace_dir = out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    step_ids = []
    pre_rewards = []
    post_rewards = []
    pre_eval_rewards = []
    post_eval_rewards = []

    for entry in trace_entries:
        step_idx = entry["step_index"]
        t_value = entry["timestep"]
        step_ids.append(step_idx)

        pre_latents = entry["pre_x0_latents_cpu"]
        post_latents = entry["post_x0_latents_cpu"]

        pre_scores, pre_eval_scores = _score_latents_in_batches(
            pipe,
            pre_latents,
            [prompt] * pre_latents.shape[0],
            steer_scorer,
            eval_scorer,
            batch_size,
            device,
            inference_dtype,
        )
        post_scores, post_eval_scores = _score_latents_in_batches(
            pipe,
            post_latents,
            [prompt] * post_latents.shape[0],
            steer_scorer,
            eval_scorer,
            batch_size,
            device,
            inference_dtype,
        )

        pre_rewards.append(pre_scores)
        post_rewards.append(post_scores)
        if eval_scorer is not None:
            pre_eval_rewards.append(pre_eval_scores)
            post_eval_rewards.append(post_eval_scores)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    steered_trace_path = out_dir / "steer_trace.csv"
    with steered_trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_index", "timestep", "pre_reward", "post_reward"])
        for idx, entry in enumerate(trace_entries):
            pre_mean = pre_rewards[idx].mean().item() if pre_rewards[idx].numel() else 0.0
            post_mean = post_rewards[idx].mean().item() if post_rewards[idx].numel() else 0.0
            writer.writerow([entry["step_index"], entry["timestep"], f"{pre_mean:.6f}", f"{post_mean:.6f}"])

    save_before_after_plot(
        step_ids,
        [scores.mean().item() for scores in pre_rewards],
        [scores.mean().item() for scores in post_rewards],
        title=f"Steer reward mean | {prompt}",
        ylabel="Reward",
        out_path=out_dir / "steer_before_after_mean.png",
    )
    save_before_after_plot(
        step_ids,
        [scores.max().item() for scores in pre_rewards],
        [scores.max().item() for scores in post_rewards],
        title=f"Steer reward max | {prompt}",
        ylabel="Reward",
        out_path=out_dir / "steer_before_after_max.png",
    )

    if eval_scorer is not None and pre_eval_rewards:
        eval_trace_path = out_dir / "eval_trace.csv"
        with eval_trace_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step_index", "timestep", "pre_eval", "post_eval"])
            for idx, entry in enumerate(trace_entries):
                pre_mean = pre_eval_rewards[idx].mean().item() if pre_eval_rewards[idx].numel() else 0.0
                post_mean = post_eval_rewards[idx].mean().item() if post_eval_rewards[idx].numel() else 0.0
                writer.writerow([entry["step_index"], entry["timestep"], f"{pre_mean:.6f}", f"{post_mean:.6f}"])

        save_before_after_plot(
            step_ids,
            [scores.mean().item() for scores in pre_eval_rewards],
            [scores.mean().item() for scores in post_eval_rewards],
            title=f"Eval reward mean | {prompt}",
            ylabel="Reward",
            out_path=out_dir / "eval_before_after_mean.png",
        )
        save_before_after_plot(
            step_ids,
            [scores.max().item() for scores in pre_eval_rewards],
            [scores.max().item() for scores in post_eval_rewards],
            title=f"Eval reward max | {prompt}",
            ylabel="Reward",
            out_path=out_dir / "eval_before_after_max.png",
        )


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
    if args.reward_guidance_rho is not None:
        config.sample.reward_guidance_rho = args.reward_guidance_rho
    if args.reward_scale_fixed is not None:
        config.sample.reward_scale_fixed = args.reward_scale_fixed
    if args.steer_start is not None:
        config.sample.steer_start = args.steer_start
    if args.steer_end is not None:
        config.sample.steer_end = args.steer_end
    if args.x0_anchor_model is not None:
        config.sample.x0_anchor_model = args.x0_anchor_model
    else:
        config.sample.x0_anchor_model = "dpm"
    if args.x0_anchor_steps is not None:
        config.sample.x0_anchor_steps = args.x0_anchor_steps
    elif config.sample.x0_anchor_model == "dpm" and config.sample.x0_anchor_steps < 2:
        config.sample.x0_anchor_steps = 2
    if args.x0_anchor_lora_path is not None:
        config.sample.x0_anchor_lora_path = args.x0_anchor_lora_path
    if args.x0_anchor_lora_scale is not None:
        config.sample.x0_anchor_lora_scale = args.x0_anchor_lora_scale

    use_reward_guidance = (
        config.reward_fn != "none"
        and config.sample.stein_loop > 0
        and config.sample.stein_step > 0
    )
    if args.offload != "none" and use_reward_guidance:
        print("[WARN] Offload is not supported with reward gradients; forcing --offload none.")
        args.offload = "none"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    inference_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if config.sample.x0_anchor_model == "dmd2" and config.sample.x0_anchor_lora_path is None:
        lora_dir = Path("models/lora/dmd2_sdxl")
        if inference_dtype == torch.float16:
            lora_file = lora_dir / "dmd2_sdxl_4step_lora_fp16.safetensors"
            lora_url = "https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora_fp16.safetensors"
        else:
            lora_file = lora_dir / "dmd2_sdxl_4step_lora.safetensors"
            lora_url = "https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora.safetensors"
        _download_lora_if_missing(lora_url, lora_file)
        config.sample.x0_anchor_lora_path = str(lora_file)

    load_kwargs = {"torch_dtype": inference_dtype, "use_safetensors": True}
    if inference_dtype == torch.float16:
        load_kwargs["variant"] = "fp16"

    pipe = DiffusionPipeline.from_pretrained(config.pretrained.model, **load_kwargs)
    if args.offload != "none" and device.type != "cuda":
        raise ValueError("CPU offload requires CUDA device.")
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
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
        reward_guidance_rho=config.sample.reward_guidance_rho,
        reward_scale_fixed=config.sample.reward_scale_fixed,
        steer_start=config.sample.steer_start,
        steer_end=config.sample.steer_end,
        intermediate_rewards=(args.save_intermediate_rewards or args.show_intermediate_rewards),
        x0_anchor_model=config.sample.x0_anchor_model,
        x0_anchor_steps=config.sample.x0_anchor_steps,
        x0_anchor_lora_path=config.sample.x0_anchor_lora_path,
        x0_anchor_lora_scale=config.sample.x0_anchor_lora_scale,
        detach_reward_anchors=args.detach_anchors,
        return_all_particles=True,
        return_dict=False,
    )
    if args.save_intermediate_images or args.save_intermediate_rewards:
        call_kwargs["callback_on_step_end"] = collect_step_latents
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    inference_start = time.time()
    with torch.no_grad():
        result = pipeline_using_approximate_sdxl(pipe, **call_kwargs)
    inference_elapsed = time.time() - inference_start

    pipeline_trace_data = None
    if isinstance(result, (tuple, list)):
        final_latents = result[0]
        if len(result) > 1:
            pipeline_trace_data = result[1]
    else:
        final_latents = result

    with torch.no_grad():
        final_images = decode_latents_sdxl(pipe, final_latents.to(device=device, dtype=inference_dtype))

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

    final_rewards_path = out_dir / "final_rewards.json"
    with final_rewards_path.open("w", encoding="utf-8") as handle:
        json.dump(final_rewards_payload, handle, indent=2)

    if args.save_intermediate_images:
        _save_intermediate_step_images(
            step_latents_for_images,
            intermediate_out_dir,
            pipe,
            steer_scorer,
            args.prompt,
            device,
            inference_dtype,
            args.trace_decode_batch_size,
            args.intermediate_max_samples,
        )

    if args.save_intermediate_rewards:
        _save_intermediate_step_rewards(
            trace_entries,
            out_dir,
            args.prompt,
            steer_scorer,
            eval_scorer,
            pipe,
            device,
            inference_dtype,
            args.trace_eval_batch,
        )

    print("\nFinal steering reward stats:")
    if final_steer_scores is not None:
        print(
            f"  mean={final_steer_scores.mean().item():.6f} "
            f"max={final_steer_scores.max().item():.6f} "
            f"min={final_steer_scores.min().item():.6f}"
        )
    else:
        print("  (no steering reward stats available)")

    if final_eval_scores is not None:
        print("\nFinal eval reward stats:")
        print(
            f"  mean={final_eval_scores.mean().item():.6f} "
            f"max={final_eval_scores.max().item():.6f} "
            f"min={final_eval_scores.min().item():.6f}"
        )

    return pipeline_trace_data


if __name__ == "__main__":
    main()

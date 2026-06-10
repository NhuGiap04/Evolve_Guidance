import argparse
import contextlib
import csv
import gc
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline

from text2img.config.sd import get_config
from text2img.diffusers_patch.pipeline_using_gradient_SD import pipeline_using_gradient_sd
from text2img.rewards import FINAL_REWARD_SCORERS, build_final_reward_scorers, build_reward_scorer


def parse_single_args():
    parser = argparse.ArgumentParser(
        description="Detailed SD Stein-guided sampling with per-step reward traces and plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pick",
        choices=["pick", "clip", "image_reward", "aesthetic", "hpsv2"],
        help="Config preset name from text2img/config/sd.py.",
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
    parser.add_argument("--stein-kernel", type=str, default=None, choices=["rbf", "rbf-full"], help="Stein kernel.")
    parser.add_argument("--stein-adagrad-eps", type=float, default=None, help="Optional AdaGrad epsilon override.")
    parser.add_argument("--kl-coeff", type=float, default=None, help="Optional reward scaling denominator override.")
    parser.add_argument(
        "--monitor-status",
        action="store_true",
        help="Print relative and absolute latent changes caused by each steered Stein step.",
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
        "--verbose",
        action="store_true",
        help="Decode and save deferred reward traces.",
    )

    return parser.parse_args()


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
    batch_size,
    device,
    inference_dtype,
):
    steer_chunks = []

    for offset in range(0, latents_cpu.shape[0], batch_size):
        chunk_cpu = latents_cpu[offset : offset + batch_size]
        with torch.inference_mode():
            chunk_latents = chunk_cpu.to(device=device, dtype=inference_dtype)
            chunk_images = decode_latents_sd(pipe, chunk_latents)
            chunk_prompts = prompts[offset : offset + chunk_images.shape[0]]
            chunk_steer = steer_scorer(chunk_images, chunk_prompts).detach().float().cpu()
            steer_chunks.append(chunk_steer)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if steer_chunks:
        return torch.cat(steer_chunks, dim=0)
    return torch.empty(0, dtype=torch.float32)


def _score_stats(scores: torch.Tensor):
    return {
        "mean": float(scores.mean().item()),
        "max": float(scores.max().item()),
        "min": float(scores.min().item()),
    }


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


def run_single_prompt(args):
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
    if args.monitor_status:
        config.sample.monitor_status = True
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
    final_scorers = build_final_reward_scorers(dtype=inference_dtype, device=device)

    out_dir = Path(args.output_dir) / f"{args.config}_seed{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    trace_storage_dtype = torch.float16 if inference_dtype == torch.float16 else torch.float32

    def collect_step_latents(_pipe, step_idx, timestep, callback_kwargs):
        if args.verbose:
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
        verbose=args.verbose,
        monitor_status=config.sample.monitor_status,
        return_all_particles=True,
        return_dict=False,
    )
    if args.verbose:
        call_kwargs["callback_on_step_end"] = collect_step_latents
        call_kwargs["callback_on_step_end_tensor_inputs"] = []

    inference_start = time.time()
    with torch.no_grad():
        result = pipeline_using_gradient_sd(pipe, **call_kwargs)
    inference_elapsed = time.time() - inference_start

    final_latents = result[0] if isinstance(result, (tuple, list)) else result
    with torch.no_grad():
        final_images = decode_latents_sd(pipe, final_latents.to(device=device, dtype=inference_dtype))

    for idx, image_tensor in enumerate(final_images):
        save_tensor_image(image_tensor, out_dir / f"sample_{idx:02d}.png")

    if device.type == "cuda":
        release_generation_modules(pipe)

    final_prompts = prompt_particles[: final_images.shape[0]]
    if len(final_prompts) != final_images.shape[0]:
        final_prompts = [args.prompt] * final_images.shape[0]

    with torch.no_grad():
        final_steer_scores = steer_scorer(final_images, final_prompts).detach().float().cpu()
        final_scores_by_scorer = {}
        for scorer_name in FINAL_REWARD_SCORERS:
            scores = final_scorers[scorer_name](final_images, final_prompts).detach().float().cpu().flatten()
            final_scores_by_scorer[scorer_name] = scores

    final_rewards_payload = {
        "prompt": args.prompt,
        "config": args.config,
        "num_images": int(final_images.shape[0]),
        "inference_time_sec": float(inference_elapsed),
        "steer_reward_name": config.reward_fn,
        "steer_rewards": [float(v) for v in final_steer_scores.tolist()],
        "steer_reward_stats": _score_stats(final_steer_scores),
        "final_particle_scorers": list(FINAL_REWARD_SCORERS),
        "final_particle_scores_by_scorer": {},
    }

    for scorer_name, scores in final_scores_by_scorer.items():
        final_rewards_payload["final_particle_scores_by_scorer"][scorer_name] = {
            "scores": [float(v) for v in scores.tolist()],
            "stats": _score_stats(scores),
        }

    with (out_dir / "final_rewards.json").open("w", encoding="utf-8") as f:
        json.dump(final_rewards_payload, f, indent=2)

    if args.verbose:
        trace_rows = []
        pre_steer_mean = []
        post_steer_mean = []
        pre_steer_max = []
        post_steer_max = []

        for trace in trace_entries:
            trace_prompts = prompt_particles[: trace["pre_x0_latents_cpu"].shape[0]]
            if len(trace_prompts) != trace["pre_x0_latents_cpu"].shape[0]:
                trace_prompts = [args.prompt] * trace["pre_x0_latents_cpu"].shape[0]

            pre_steer_scores = _score_latents_in_batches(
                pipe=pipe,
                latents_cpu=trace["pre_x0_latents_cpu"],
                prompts=trace_prompts,
                steer_scorer=steer_scorer,
                batch_size=max(1, int(config.sample.num_particles)),
                device=device,
                inference_dtype=inference_dtype,
            )
            post_steer_scores = _score_latents_in_batches(
                pipe=pipe,
                latents_cpu=trace["post_x0_latents_cpu"],
                prompts=trace_prompts,
                steer_scorer=steer_scorer,
                batch_size=max(1, int(config.sample.num_particles)),
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

    print("Saved outputs to:", out_dir)
    print(f"Inference time (pipeline only): {inference_elapsed:.4f}s")
    print(
        "Final steering reward stats: "
        f"mean={final_steer_scores.mean().item():.6f} max={final_steer_scores.max().item():.6f}"
    )
    for scorer_name in FINAL_REWARD_SCORERS:
        stats = final_rewards_payload["final_particle_scores_by_scorer"][scorer_name]["stats"]
        print(f"Final {scorer_name} stats: mean={stats['mean']:.6f} max={stats['max']:.6f}")


def single_main() -> None:
    run_single_prompt(parse_single_args())


FINAL_SCORERS = ("clip", "pick", "image_reward", "aesthetic", "hpsv2")


def _supports_color() -> bool:
    return sys.stdout.isatty()


class _Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


USE_COLOR = _supports_color()


def _c(text: str, *styles: str) -> str:
    if not USE_COLOR or not styles:
        return text
    return "".join(styles) + text + _Style.RESET


def _title(text: str) -> None:
    line = "=" * len(text)
    print(_c(line, _Style.CYAN, _Style.BOLD))
    print(_c(text, _Style.CYAN, _Style.BOLD))
    print(_c(line, _Style.CYAN, _Style.BOLD))


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    if not slug:
        return "prompt"
    return slug[:max_len].rstrip("-")


def _truncate(text: str, max_len: int = 72) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_prompts_from_json(data: Any) -> List[str]:
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return [item.strip() for item in data if item.strip()]

        prompts: List[str] = []
        for item in data:
            if isinstance(item, str):
                prompt = item.strip()
                if prompt:
                    prompts.append(prompt)
                continue

            if isinstance(item, dict):
                value = item.get("prompt")
                if isinstance(value, str):
                    prompt = value.strip()
                    if prompt:
                        prompts.append(prompt)
                    continue

                value = item.get("text")
                if isinstance(value, str):
                    prompt = value.strip()
                    if prompt:
                        prompts.append(prompt)
                    continue

                raise ValueError("JSON list entries must be strings or objects with 'prompt'/'text'.")

            raise ValueError("Unsupported JSON list entry type for prompts.")

        return prompts

    if isinstance(data, dict):
        if "prompts" in data:
            return _extract_prompts_from_json(data["prompts"])
        raise ValueError("JSON object must contain a 'prompts' field.")

    raise ValueError("JSON prompt file must be a list or an object with 'prompts'.")


def load_prompts(file_path: Path) -> List[str]:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        prompts = []
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
        return prompts

    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return _extract_prompts_from_json(data)

    raise ValueError(f"Unsupported prompt file extension: {suffix}. Use .txt or .json")


def _build_single_prompt_args(args: argparse.Namespace, prompt: str, run_output_dir: Path) -> argparse.Namespace:
    values = {
        "prompt": prompt,
        "output_dir": str(run_output_dir),
        "config": args.config,
        "negative_prompt": args.negative_prompt,
        "device": args.device,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "num_particles": args.num_particles,
        "batch_p": args.batch_p,
        "stein_step": args.stein_step,
        "stein_loop": args.stein_loop,
        "stein_kernel": args.stein_kernel,
        "stein_adagrad_eps": args.stein_adagrad_eps,
        "kl_coeff": args.kl_coeff,
        "monitor_status": args.monitor_status,
        "steer_start": args.steer_start,
        "steer_end": args.steer_end,
        "verbose": args.verbose,
    }
    return argparse.Namespace(**values)


def _single_prompt_call_description(single_args: argparse.Namespace) -> str:
    return (
        "run_single_prompt("
        f"prompt={single_args.prompt!r}, "
        f"output_dir={single_args.output_dir!r}, "
        f"config={single_args.config!r}, "
        f"device={single_args.device!r}"
        ")"
    )


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    print()
    _title("Batch Summary")

    headers = ["Idx", "Status", "Time(s)", "Prompt"]
    table_data = []
    for row in rows:
        table_data.append(
            [
                str(row["index"]),
                row["status"],
                f"{row['elapsed']:.2f}",
                _truncate(row["prompt"], 68),
            ]
        )

    col_widths = [len(h) for h in headers]
    for r in table_data:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def format_row(cells: List[str]) -> str:
        padded = [cells[i].ljust(col_widths[i]) for i in range(len(cells))]
        return " | ".join(padded)

    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in table_data:
        status = r[1]
        if status == "OK":
            r[1] = _c(status, _Style.GREEN, _Style.BOLD)
        elif status == "DRY":
            r[1] = _c(status, _Style.YELLOW, _Style.BOLD)
        else:
            r[1] = _c(status, _Style.RED, _Style.BOLD)
        print(format_row(r))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grad_sd.py over one prompt or prompts from a .txt/.json file.",
    )
    parser.add_argument("--prompts-file", type=Path, default=None, help="Path to .txt or .json prompts file.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A close up of a handpalm with leaves growing from it.",
        help="Prompt used when --prompts-file is omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pick",
        choices=["pick", "clip", "image_reward", "aesthetic", "hpsv2"],
    )
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/sd_batch"))
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)

    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--batch-p", type=int, default=None)
    parser.add_argument("--stein-step", type=float, default=None)
    parser.add_argument("--stein-loop", type=int, default=None)
    parser.add_argument("--stein-kernel", type=str, default=None, choices=["rbf", "rbf-full"])
    parser.add_argument("--stein-adagrad-eps", type=float, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)
    parser.add_argument("--monitor-status", action="store_true")
    parser.add_argument("--steer-start", type=int, default=None)
    parser.add_argument("--steer-end", type=int, default=None)

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--start-index", type=int, default=0, help="Start from this 0-based prompt index.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts to run.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first failed prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where to save per-run stdout/stderr logs (default: <output-dir>/_batch_logs).",
    )

    return parser.parse_args()


def _to_metric_str(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


def _blank_reward_stats() -> Dict[str, str]:
    stats = {"steer_mean": "", "steer_max": ""}
    for scorer_name in FINAL_SCORERS:
        stats[f"{scorer_name}_mean"] = ""
        stats[f"{scorer_name}_max"] = ""
    return stats


def _final_rewards_candidates(run_output_dir: Path) -> List[Path]:
    candidates = [run_output_dir / "final_rewards.json"]
    if run_output_dir.exists():
        candidates.extend(sorted(run_output_dir.glob("*/final_rewards.json")))
        candidates.extend(sorted(run_output_dir.rglob("final_rewards.json")))

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)
    return unique_candidates


def _load_final_rewards_payload(run_output_dir: Path) -> Optional[Dict[str, Any]]:
    for final_rewards_path in _final_rewards_candidates(run_output_dir):
        if not final_rewards_path.exists():
            continue
        try:
            payload = json.loads(final_rewards_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_reward_stats_from_stdout(stdout: str, stats: Dict[str, str]) -> None:
    steer_match = re.search(
        r"Final steering reward stats:\s*mean=([-+0-9.eE]+)\s+max=([-+0-9.eE]+)",
        stdout,
    )
    if steer_match:
        if not stats["steer_mean"]:
            stats["steer_mean"] = _to_metric_str(float(steer_match.group(1)))
        if not stats["steer_max"]:
            stats["steer_max"] = _to_metric_str(float(steer_match.group(2)))

    for match in re.finditer(
        r"Final ([A-Za-z0-9_]+) stats:\s*mean=([-+0-9.eE]+)\s+max=([-+0-9.eE]+)",
        stdout,
    ):
        scorer_name = match.group(1)
        if scorer_name not in FINAL_SCORERS:
            continue
        mean_key = f"{scorer_name}_mean"
        max_key = f"{scorer_name}_max"
        if not stats[mean_key]:
            stats[mean_key] = _to_metric_str(float(match.group(2)))
        if not stats[max_key]:
            stats[max_key] = _to_metric_str(float(match.group(3)))


def _extract_reward_stats(run_output_dir: Path, stdout: str = "") -> Dict[str, str]:
    stats = _blank_reward_stats()
    payload = _load_final_rewards_payload(run_output_dir)

    if payload is None:
        _extract_reward_stats_from_stdout(stdout, stats)
        return stats

    steer_stats = payload.get("steer_reward_stats", {})
    stats["steer_mean"] = _to_metric_str(steer_stats.get("mean"))
    stats["steer_max"] = _to_metric_str(steer_stats.get("max"))

    scorer_payload = payload.get("final_particle_scores_by_scorer", {})
    for scorer_name in FINAL_SCORERS:
        scorer_stats = scorer_payload.get(scorer_name, {}).get("stats", {})
        stats[f"{scorer_name}_mean"] = _to_metric_str(scorer_stats.get("mean"))
        stats[f"{scorer_name}_max"] = _to_metric_str(scorer_stats.get("max"))

    _extract_reward_stats_from_stdout(stdout, stats)
    return stats


def _fmt_reward_stat(value: str) -> str:
    return value if value else "NA"


def _build_eval_row(
    global_idx: int,
    prompt: str,
    status: str,
    reward_stats: Dict[str, str],
) -> Dict[str, str]:
    row = {
        "index": str(global_idx),
        "prompt": prompt,
        "status": status,
        "steer_mean": reward_stats["steer_mean"],
        "steer_max": reward_stats["steer_max"],
    }
    for scorer_name in FINAL_SCORERS:
        row[f"{scorer_name}_mean"] = reward_stats[f"{scorer_name}_mean"]
        row[f"{scorer_name}_max"] = reward_stats[f"{scorer_name}_max"]
    return row


def _reward_log_line(reward_stats: Dict[str, str]) -> str:
    parts = [
        f"steer_mean={_fmt_reward_stat(reward_stats['steer_mean'])}",
        f"steer_max={_fmt_reward_stat(reward_stats['steer_max'])}",
    ]
    for scorer_name in FINAL_SCORERS:
        parts.append(f"{scorer_name}_mean={_fmt_reward_stat(reward_stats[f'{scorer_name}_mean'])}")
    return "  rewards: " + " ".join(parts)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _resolve_pipeline_config(args: argparse.Namespace) -> Dict[str, Any]:
    # Prefer the source-of-truth config module when available.
    try:
        from text2img.config.sd import get_config as get_sd_config  # type: ignore

        config = get_sd_config(args.config)
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
        if args.monitor_status:
            config.sample.monitor_status = True
        if args.steer_start is not None:
            config.sample.steer_start = args.steer_start
        if args.steer_end is not None:
            config.sample.steer_end = args.steer_end

        model = str(config.pretrained.model)
        model_revision = str(getattr(config.pretrained, "revision", ""))
        reward_fn = str(config.reward_fn)
        seed = int(config.seed)
        sample = _json_safe(config.sample.to_dict())
    except Exception:
        # Fallback for environments without config dependencies.
        model = "runwayml/stable-diffusion-v1-5"
        model_revision = "main"
        reward_fn = args.config
        seed = int(args.seed) if args.seed is not None else 42
        sample = {
            "num_steps": 100,
            "eta": 1.0,
            "guidance_scale": 5.0,
            "batch_size": 1,
            "num_particles": 4,
            "batch_p": 1,
            "stein_step": 0.02,
            "stein_loop": 2,
            "stein_kernel": "rbf",
            "stein_adagrad_eps": 1e-8,
            "stein_adagrad_clip": None,
            "kl_coeff": 0.0001,
            "steer_start": None,
            "steer_end": None,
            "monitor_status": False,
        }
        if args.num_steps is not None:
            sample["num_steps"] = args.num_steps
        if args.batch_size is not None:
            sample["batch_size"] = args.batch_size
        if args.guidance_scale is not None:
            sample["guidance_scale"] = args.guidance_scale
        if args.eta is not None:
            sample["eta"] = args.eta
        if args.num_particles is not None:
            sample["num_particles"] = args.num_particles
        if args.batch_p is not None:
            sample["batch_p"] = args.batch_p
        if args.stein_step is not None:
            sample["stein_step"] = args.stein_step
        if args.stein_loop is not None:
            sample["stein_loop"] = args.stein_loop
        if args.stein_kernel is not None:
            sample["stein_kernel"] = args.stein_kernel
        if args.stein_adagrad_eps is not None:
            sample["stein_adagrad_eps"] = args.stein_adagrad_eps
        if args.kl_coeff is not None:
            sample["kl_coeff"] = args.kl_coeff
        if args.monitor_status:
            sample["monitor_status"] = True
        if args.steer_start is not None:
            sample["steer_start"] = args.steer_start
        if args.steer_end is not None:
            sample["steer_end"] = args.steer_end

    payload = {
        "runner": "gradient_sd_batch",
        "created_at_unix": float(time.time()),
        "pipeline_type": "sd",
        "pipeline_script": str(Path(__file__)),
        "config_name": args.config,
        "model": model,
        "model_revision": model_revision,
        "reward_fn": reward_fn,
        "seed": seed,
        "device": args.device,
        "negative_prompt": args.negative_prompt,
        "sample": _json_safe(sample),
        "batch_args": {k: _json_safe(v) for k, v in vars(args).items()},
    }
    return payload


def main() -> int:
    args = parse_args()

    single_prompt_mode = args.prompts_file is None
    if single_prompt_mode:
        prompts = [args.prompt]
    else:
        if not args.prompts_file.exists():
            print(_c(f"Prompt file not found: {args.prompts_file}", _Style.RED, _Style.BOLD))
            return 2
        prompts = load_prompts(args.prompts_file)
        if not prompts:
            print(_c("No prompts were loaded from input file.", _Style.RED, _Style.BOLD))
            return 2

    if args.start_index < 0 or args.start_index >= len(prompts):
        print(_c("--start-index is out of range.", _Style.RED, _Style.BOLD))
        return 2

    end_index = len(prompts)
    if args.max_prompts is not None:
        if args.max_prompts < 1:
            print(_c("--max-prompts must be >= 1.", _Style.RED, _Style.BOLD))
            return 2
        end_index = min(end_index, args.start_index + args.max_prompts)

    selected_prompts = prompts[args.start_index:end_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (args.output_dir / "_batch_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config_payload = _resolve_pipeline_config(args)
    pipeline_config_path = args.output_dir / "pipeline_run_config.json"
    with pipeline_config_path.open("w", encoding="utf-8") as f:
        json.dump(pipeline_config_payload, f, indent=2)

    _title("Gradient SD Batch Runner")
    print(f"Prompt file : {args.prompts_file if args.prompts_file is not None else '<single prompt>'}")
    print(f"Script      : {Path(__file__)}")
    print(f"Runs        : {len(selected_prompts)} (from index {args.start_index})")
    print(f"Output root : {args.output_dir}")
    print(f"Log dir     : {log_dir}")
    print()

    rows: List[Dict[str, Any]] = []
    success_count = 0
    csv_path = args.output_dir / "batch_eval_summary.csv"
    csv_fieldnames = ["index", "prompt", "steer_mean", "steer_max"]
    for scorer_name in FINAL_SCORERS:
        csv_fieldnames.extend([f"{scorer_name}_mean", f"{scorer_name}_max"])
    csv_fieldnames.append("status")

    batch_start = time.time()
    total_runs = len(selected_prompts)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        eval_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        eval_writer.writeheader()
        csv_file.flush()
        os.fsync(csv_file.fileno())

        for local_idx, prompt in enumerate(selected_prompts, start=1):
            global_idx = args.start_index + local_idx - 1
            prompt_slug = _slugify(prompt)
            run_name = f"run_{global_idx:04d}_{prompt_slug}"
            run_output_dir = args.output_dir if single_prompt_mode else args.output_dir / run_name

            single_args = _build_single_prompt_args(args, prompt, run_output_dir)

            print(_c(f"[{local_idx:03d}/{total_runs:03d}]", _Style.BOLD), _truncate(prompt, 100))
            print(_c("  output:", _Style.DIM), run_output_dir)

            if args.dry_run:
                success_count += 1
                print(_c("  dry-run call:", _Style.DIM), _single_prompt_call_description(single_args))
                rows.append(
                    {
                        "index": global_idx,
                        "status": "DRY",
                        "elapsed": 0.0,
                        "prompt": prompt,
                    }
                )
                reward_stats = _blank_reward_stats()
                eval_row = _build_eval_row(global_idx, prompt, "DRY", reward_stats)
                eval_writer.writerow(eval_row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                continue

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            start = time.time()
            returncode = 0
            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    run_single_prompt(single_args)
            except Exception:
                returncode = 1
                traceback.print_exc(file=stderr_buffer)
            elapsed = time.time() - start

            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            stdout_path = log_dir / f"{run_name}.stdout.log"
            stderr_path = log_dir / f"{run_name}.stderr.log"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

            reward_stats = _extract_reward_stats(run_output_dir, stdout)
            if returncode == 0:
                success_count += 1
                status = _c("OK", _Style.GREEN, _Style.BOLD)
                rows.append(
                    {
                        "index": global_idx,
                        "status": "OK",
                        "elapsed": elapsed,
                        "prompt": prompt,
                    }
                )
                eval_row = _build_eval_row(global_idx, prompt, "OK", reward_stats)
                eval_writer.writerow(eval_row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                print(_c(f"  status: {status}  time: {elapsed:.2f}s", _Style.DIM))
                print(_c(_reward_log_line(reward_stats), _Style.DIM))
            else:
                status = _c("FAIL", _Style.RED, _Style.BOLD)
                rows.append(
                    {
                        "index": global_idx,
                        "status": "FAIL",
                        "elapsed": elapsed,
                        "prompt": prompt,
                    }
                )
                eval_row = _build_eval_row(global_idx, prompt, "FAIL", reward_stats)
                eval_writer.writerow(eval_row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                print(_c(f"  status: {status}  time: {elapsed:.2f}s  code: {returncode}", _Style.DIM))
                print(_c(_reward_log_line(reward_stats), _Style.DIM))
                stderr_tail = stderr.splitlines()[-20:]
                if stderr_tail:
                    print(_c("  stderr tail:", _Style.YELLOW, _Style.BOLD))
                    for line in stderr_tail:
                        print("   ", line)

                if args.stop_on_error:
                    print(_c("Stopping due to --stop-on-error", _Style.YELLOW, _Style.BOLD))
                    break

            print()

    total_elapsed = time.time() - batch_start
    _print_summary(rows)

    print()
    _title("Result")
    print(f"Succeeded : {success_count}/{len(rows)}")
    print(f"Failed    : {len(rows) - success_count}")
    print(f"Wall time : {total_elapsed:.2f}s")
    print(f"Logs      : {log_dir}")
    print(f"Eval CSV  : {csv_path}")

    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

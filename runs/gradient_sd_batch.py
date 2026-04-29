#!/usr/bin/env python3
"""Batch runner for runs/single/gradient_sd.py using prompts from .txt or .json."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _append_optional_arg(cmd: List[str], flag: str, value: Optional[Any]) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _build_sd_cmd(args: argparse.Namespace, prompt: str, run_output_dir: Path) -> List[str]:
    cmd = [
        args.python,
        str(args.sd_script),
        "--config",
        args.config,
        "--prompt",
        prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--output-dir",
        str(run_output_dir),
        "--device",
        args.device,
    ]

    _append_optional_arg(cmd, "--seed", args.seed)
    _append_optional_arg(cmd, "--num-steps", args.num_steps)
    _append_optional_arg(cmd, "--batch-size", args.batch_size)
    _append_optional_arg(cmd, "--guidance-scale", args.guidance_scale)
    _append_optional_arg(cmd, "--eta", args.eta)

    _append_optional_arg(cmd, "--num-particles", args.num_particles)
    _append_optional_arg(cmd, "--batch-p", args.batch_p)
    _append_optional_arg(cmd, "--stein-step", args.stein_step)
    _append_optional_arg(cmd, "--stein-loop", args.stein_loop)
    _append_optional_arg(cmd, "--stein-kernel", args.stein_kernel)
    _append_optional_arg(cmd, "--stein-adagrad-eps", args.stein_adagrad_eps)
    _append_optional_arg(cmd, "--kl-coeff", args.kl_coeff)
    _append_optional_arg(cmd, "--steer-start", args.steer_start)
    _append_optional_arg(cmd, "--steer-end", args.steer_end)
    if args.monitor_status:
        cmd.append("--monitor-status")

    if args.verbose:
        cmd.append("--verbose")
        _append_optional_arg(cmd, "--intermediate-max-samples", args.intermediate_max_samples)
        _append_optional_arg(cmd, "--trace-eval-batch", args.trace_eval_batch)

    return cmd


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
        description="Run gradient_sd.py over prompts from a .txt or .json file.",
    )
    parser.add_argument("--prompts-file", type=Path, required=True, help="Path to .txt or .json prompts file.")
    parser.add_argument(
        "--gradient-script",
        "--sd-script",
        dest="sd_script",
        type=Path,
        default=Path("runs/single/gradient_sd.py"),
        help="Path to the single-prompt gradient SD script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch gradient_sd.py.",
    )
    parser.add_argument("--config", type=str, default="pick", choices=["pick", "clip", "seg"])
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
    parser.add_argument("--stein-kernel", type=str, default=None, choices=["rbf"])
    parser.add_argument("--stein-adagrad-eps", type=float, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)
    parser.add_argument("--monitor-status", action="store_true")
    parser.add_argument("--steer-start", type=int, default=None)
    parser.add_argument("--steer-end", type=int, default=None)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace-eval-batch", type=int, default=None)
    parser.add_argument("--intermediate-max-samples", type=int, default=None)

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
        from config.sd import get_config as get_sd_config  # type: ignore

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
        reward_fn = {"pick": "pick", "clip": "clip", "seg": "clip"}.get(args.config, args.config)
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
        "pipeline_script": str(args.sd_script),
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

    if not args.prompts_file.exists():
        print(_c(f"Prompt file not found: {args.prompts_file}", _Style.RED, _Style.BOLD))
        return 2
    if not args.sd_script.exists():
        print(_c(f"Gradient SD script not found: {args.sd_script}", _Style.RED, _Style.BOLD))
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
    print(f"Prompt file : {args.prompts_file}")
    print(f"Script      : {args.sd_script}")
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
            run_output_dir = args.output_dir / run_name

            cmd = _build_sd_cmd(args, prompt, run_output_dir)

            print(_c(f"[{local_idx:03d}/{total_runs:03d}]", _Style.BOLD), _truncate(prompt, 100))
            print(_c("  output:", _Style.DIM), run_output_dir)

            if args.dry_run:
                success_count += 1
                print(_c("  dry-run command:", _Style.DIM), " ".join(cmd))
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

            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start

            stdout_path = log_dir / f"{run_name}.stdout.log"
            stderr_path = log_dir / f"{run_name}.stderr.log"
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")

            reward_stats = _extract_reward_stats(run_output_dir, proc.stdout or "")
            if proc.returncode == 0:
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
                print(_c(f"  status: {status}  time: {elapsed:.2f}s  code: {proc.returncode}", _Style.DIM))
                print(_c(_reward_log_line(reward_stats), _Style.DIM))
                stderr_tail = (proc.stderr or "").splitlines()[-20:]
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

#!/usr/bin/env python3
"""Batch runner for runs/single/gradient_sd.py using prompts from .txt or .json."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import matplotlib.pyplot as plt


PLOT_LOCK = threading.Lock()


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


def _tail_lines(text: str, count: int) -> List[str]:
    if count <= 0 or not text:
        return []
    return text.splitlines()[-count:]


def _extract_intermediate_reward_trace(stdout: str) -> Dict[str, List[float]]:
    pattern = re.compile(
        r"^\[t=(\d+)\]\s+pre_mean=([\-0-9.eE]+)\s+pre_max=([\-0-9.eE]+)\s+"
        r"post_mean=([\-0-9.eE]+)\s+post_max=([\-0-9.eE]+)$"
    )
    trace = {
        "timestep": [],
        "pre_mean": [],
        "post_mean": [],
        "pre_max": [],
        "post_max": [],
    }

    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        trace["timestep"].append(float(match.group(1)))
        trace["pre_mean"].append(float(match.group(2)))
        trace["pre_max"].append(float(match.group(3)))
        trace["post_mean"].append(float(match.group(4)))
        trace["post_max"].append(float(match.group(5)))

    return trace


def _plot_reward_trace(prompt: str, trace: Dict[str, List[float]], run_name: str, device: str) -> None:
    if not trace["timestep"]:
        return

    x = list(range(1, len(trace["timestep"]) + 1))
    with PLOT_LOCK:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"{run_name} | {device}\\n{_truncate(prompt, 120)}")

        axes[0].plot(x, trace["pre_mean"], label="before_stein", linewidth=2)
        axes[0].plot(x, trace["post_mean"], label="after_stein", linewidth=2)
        axes[0].set_title("Reward Mean")
        axes[0].set_xlabel("Steered step")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(x, trace["pre_max"], label="before_stein", linewidth=2)
        axes[1].plot(x, trace["post_max"], label="after_stein", linewidth=2)
        axes[1].set_title("Reward Max")
        axes[1].set_xlabel("Steered step")
        axes[1].set_ylabel("Reward")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


def _build_sd_cmd(args: argparse.Namespace, prompt: str, run_output_dir: Path, device: str) -> List[str]:
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
        "--eval-reward",
        args.eval_reward,
        "--device",
        device,
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

    if args.save_intermediate_images:
        cmd.append("--save-intermediate-images")
        _append_optional_arg(cmd, "--trace-decode-batch-size", args.trace_decode_batch_size)
        _append_optional_arg(cmd, "--intermediate-max-samples", args.intermediate_max_samples)
    if args.save_intermediate_rewards:
        cmd.append("--save-intermediate-rewards")
    if args.plot_after_run:
        cmd.append("--show-intermediate-rewards")
    _append_optional_arg(cmd, "--trace-eval-batch", args.trace_eval_batch)

    return cmd


def _split_prompts_across_devices(prompts: List[str], devices: List[str]) -> List[List[Tuple[int, str]]]:
    if not devices:
        raise ValueError("At least one device must be provided.")

    shards: List[List[Tuple[int, str]]] = [[] for _ in range(len(devices))]
    for index, prompt in enumerate(prompts):
        shards[index % len(devices)].append((index, prompt))
    return shards


def _run_prompt_shard(
    args: argparse.Namespace,
    shard_prompts: List[Tuple[int, str]],
    device: str,
    log_dir: Path,
    repo_root: Path,
    launch_env: Dict[str, str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    success_count = 0

    for local_idx, (global_idx, prompt) in enumerate(shard_prompts, start=1):
        prompt_slug = _slugify(prompt)
        run_name = f"run_{global_idx:04d}_{prompt_slug}"
        run_output_dir = args.output_dir / run_name

        cmd = _build_sd_cmd(args, prompt, run_output_dir, device)

        print(_c(f"[{device}] [{local_idx:03d}/{len(shard_prompts):03d}]", _Style.BOLD), _truncate(prompt, 100))
        print(_c("  output:", _Style.DIM), run_output_dir)
        if args.verbose:
            print(_c("  command:", _Style.DIM), " ".join(cmd))

        if args.dry_run:
            success_count += 1
            print(_c("  dry-run command:", _Style.DIM), " ".join(cmd))
            rows.append({"index": global_idx, "status": "DRY", "elapsed": 0.0, "prompt": prompt})
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "steer_mean": "",
                "steer_max": "",
                "eval_mean": "",
                "eval_max": "",
                "status": "DRY",
            })
            continue

        start = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env=launch_env,
        )
        elapsed = time.time() - start

        stdout_path = log_dir / f"{run_name}.stdout.log"
        stderr_path = log_dir / f"{run_name}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        if args.verbose:
            print(_c("  stdout log:", _Style.DIM), stdout_path)
            print(_c("  stderr log:", _Style.DIM), stderr_path)

            stdout_tail = _tail_lines(proc.stdout or "", args.verbose_tail_lines)
            stderr_tail = _tail_lines(proc.stderr or "", args.verbose_tail_lines)

            if stdout_tail:
                print(_c(f"  stdout tail ({len(stdout_tail)} lines):", _Style.CYAN, _Style.BOLD))
                for line in stdout_tail:
                    print("   ", line)

            if stderr_tail:
                print(_c(f"  stderr tail ({len(stderr_tail)} lines):", _Style.YELLOW, _Style.BOLD))
                for line in stderr_tail:
                    print("   ", line)

        reward_stats = _extract_reward_stats(proc.stdout or "")
        trace_data = _extract_intermediate_reward_trace(proc.stdout or "")
        if proc.returncode == 0:
            success_count += 1
            rows.append({"index": global_idx, "status": "OK", "elapsed": elapsed, "prompt": prompt})
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "steer_mean": reward_stats["steer_mean"],
                "steer_max": reward_stats["steer_max"],
                "eval_mean": reward_stats["eval_mean"],
                "eval_max": reward_stats["eval_max"],
                "status": "OK",
            })
            print(_c(f"  status: {_c('OK', _Style.GREEN, _Style.BOLD)}  time: {elapsed:.2f}s", _Style.DIM))
            if args.plot_after_run:
                if trace_data["timestep"]:
                    _plot_reward_trace(prompt=prompt, trace=trace_data, run_name=run_name, device=device)
                else:
                    print(_c("  plot: no intermediate reward trace found in stdout.", _Style.YELLOW, _Style.BOLD))
        else:
            rows.append({"index": global_idx, "status": "FAIL", "elapsed": elapsed, "prompt": prompt})
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "steer_mean": reward_stats["steer_mean"],
                "steer_max": reward_stats["steer_max"],
                "eval_mean": reward_stats["eval_mean"],
                "eval_max": reward_stats["eval_max"],
                "status": "FAIL",
            })
            print(_c(f"  status: {_c('FAIL', _Style.RED, _Style.BOLD)}  time: {elapsed:.2f}s  code: {proc.returncode}", _Style.DIM))
            stderr_tail = (proc.stderr or "").splitlines()[-20:]
            if stderr_tail:
                print(_c("  stderr tail:", _Style.YELLOW, _Style.BOLD))
                for line in stderr_tail:
                    print("   ", line)
            if args.stop_on_error:
                raise RuntimeError(f"Stopping due to --stop-on-error on prompt index {global_idx}")

        print()

    return rows, eval_rows, success_count


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
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="image_reward",
        choices=["none", "clip", "pick", "image_reward"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional list of GPU devices to shard prompts across, e.g. --devices cuda:0 cuda:1.",
    )
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
    parser.add_argument("--steer-start", type=int, default=None)
    parser.add_argument("--steer-end", type=int, default=None)

    parser.add_argument("--save-intermediate-images", action="store_true")
    parser.add_argument("--save-intermediate-rewards", action="store_true")
    parser.add_argument(
        "--plot-after-run",
        action="store_true",
        default=True,
        help="Display a before/after Stein reward plot after each completed run.",
    )
    parser.add_argument(
        "--no-plot-after-run",
        dest="plot_after_run",
        action="store_false",
        help="Disable per-run reward plotting.",
    )
    parser.add_argument("--trace-decode-batch-size", type=int, default=None)
    parser.add_argument("--trace-eval-batch", type=int, default=None)
    parser.add_argument("--intermediate-max-samples", type=int, default=None)

    parser.add_argument("--start-index", type=int, default=0, help="Start from this 0-based prompt index.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts to run.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first failed prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed per-run diagnostics (enabled by default).",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose per-run diagnostics.",
    )
    parser.add_argument(
        "--verbose-tail-lines",
        type=int,
        default=20,
        help="How many stdout/stderr tail lines to print when --verbose is enabled.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where to save per-run stdout/stderr logs (default: <output-dir>/_batch_logs).",
    )

    return parser.parse_args()


def _extract_reward_stats(stdout: str) -> Dict[str, str]:
    """Parse final steering/eval reward stats from gradient_sd.py stdout."""
    # Looks for:
    # - Final eval reward stats (...): mean=-0.330566 max=-0.330566
    # - Final steering reward stats: mean=-0.330566 max=-0.330566
    result = {
        "steer_mean": "",
        "steer_max": "",
        "eval_mean": "",
        "eval_max": "",
    }
    for line in stdout.splitlines():
        if "Final eval reward stats" in line:
            # Extract mean and max using regex
            import re
            m = re.search(r"mean=([\-0-9.eE]+) max=([\-0-9.eE]+)", line)
            if m:
                result["eval_mean"] = m.group(1)
                result["eval_max"] = m.group(2)
        elif "Final steering reward stats" in line:
            import re
            m = re.search(r"mean=([\-0-9.eE]+) max=([\-0-9.eE]+)", line)
            if m:
                result["steer_mean"] = m.group(1)
                result["steer_max"] = m.group(2)
    return result

def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if not args.prompts_file.is_absolute() and not args.prompts_file.exists():
        args.prompts_file = (repo_root / args.prompts_file).resolve()
    if not args.sd_script.is_absolute() and not args.sd_script.exists():
        args.sd_script = (repo_root / args.sd_script).resolve()

    launch_env = os.environ.copy()
    existing_pythonpath = launch_env.get("PYTHONPATH", "")
    repo_root_str = str(repo_root)
    launch_env["PYTHONPATH"] = (
        repo_root_str if not existing_pythonpath else f"{repo_root_str}{os.pathsep}{existing_pythonpath}"
    )

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
    devices = args.devices if args.devices is not None else [args.device]

    if not args.output_dir.is_absolute():
        args.output_dir = (repo_root / args.output_dir).resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir or (args.output_dir / "_batch_logs")
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    _title("Gradient SD Batch Runner")
    print(f"Prompt file : {args.prompts_file}")
    print(f"Script      : {args.sd_script}")
    print(f"Runs        : {len(selected_prompts)} (from index {args.start_index})")
    print(f"Output root : {args.output_dir}")
    print(f"Log dir     : {log_dir}")
    print(f"Devices     : {', '.join(devices)}")
    print()

    rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    success_count = 0

    batch_start = time.time()
    shards = _split_prompts_across_devices(selected_prompts, devices)
    shard_inputs = [
        [(args.start_index + prompt_index, prompt) for prompt_index, prompt in shard]
        for shard in shards
        if shard
    ]

    if len(devices) == 1:
        shard_rows, shard_eval_rows, shard_success = _run_prompt_shard(
            args,
            shard_inputs[0],
            devices[0],
            log_dir,
            repo_root,
            launch_env,
        )
        rows.extend(shard_rows)
        eval_rows.extend(shard_eval_rows)
        success_count += shard_success
    else:
        with ThreadPoolExecutor(max_workers=len(shard_inputs)) as executor:
            futures = [
                executor.submit(_run_prompt_shard, args, shard, devices[i], log_dir, repo_root, launch_env)
                for i, shard in enumerate(shard_inputs)
            ]
            for future in as_completed(futures):
                shard_rows, shard_eval_rows, shard_success = future.result()
                rows.extend(shard_rows)
                eval_rows.extend(shard_eval_rows)
                success_count += shard_success

    rows.sort(key=lambda row: row["index"])
    eval_rows.sort(key=lambda row: row["index"])

    total_elapsed = time.time() - batch_start
    _print_summary(rows)

    # Save eval summary CSV
    csv_path = args.output_dir / "batch_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "prompt",
                "steer_mean",
                "steer_max",
                "eval_mean",
                "eval_max",
                "status",
            ],
        )
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(row)

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

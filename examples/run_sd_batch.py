#!/usr/bin/env python3
"""Batch runner for examples/sd.py using prompts from .txt or .json."""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        "--eval-reward",
        args.eval_reward,
        "--device",
        args.device,
    ]

    _append_optional_arg(cmd, "--seed", args.seed)
    _append_optional_arg(cmd, "--num-steps", args.num_steps)
    _append_optional_arg(cmd, "--batch-size", args.batch_size)
    _append_optional_arg(cmd, "--guidance-scale", args.guidance_scale)
    _append_optional_arg(cmd, "--eta", args.eta)
    _append_optional_arg(cmd, "--lpips-ref-dir", args.lpips_ref_dir)
    _append_optional_arg(cmd, "--lpips-net", args.lpips_net)

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
        else:
            r[1] = _c(status, _Style.RED, _Style.BOLD)
        print(format_row(r))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run examples/sd.py over prompts from a .txt or .json file.",
    )
    parser.add_argument("--prompts-file", type=Path, required=True, help="Path to .txt or .json prompts file.")
    parser.add_argument(
        "--sd-script",
        type=Path,
        default=Path("examples/sd.py"),
        help="Path to the single-prompt SD script.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch sd.py.")
    parser.add_argument("--config", type=str, default="pick", choices=["pick", "clip", "seg"])
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/sd_batch"))
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="image_reward",
        choices=["none", "clip", "pick", "image_reward", "lpips"],
    )
    parser.add_argument("--lpips-ref-dir", type=str, default=None)
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)

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


def _extract_eval_stats(stdout: str) -> Dict[str, str]:
    """Parse final eval reward stats from sd.py stdout."""
    result = {"mean": "", "max": ""}
    for line in stdout.splitlines():
        if "Final eval reward stats" in line:
            match = re.search(r"mean=([\-0-9.eE]+) max=([\-0-9.eE]+)", line)
            if match:
                result["mean"] = match.group(1)
                result["max"] = match.group(2)
            break
    return result


def main() -> int:
    args = parse_args()

    if not args.prompts_file.exists():
        print(_c(f"Prompt file not found: {args.prompts_file}", _Style.RED, _Style.BOLD))
        return 2
    if not args.sd_script.exists():
        print(_c(f"SD script not found: {args.sd_script}", _Style.RED, _Style.BOLD))
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

    _title("SD Batch Runner")
    print(f"Prompt file : {args.prompts_file}")
    print(f"Script      : {args.sd_script}")
    print(f"Runs        : {len(selected_prompts)} (from index {args.start_index})")
    print(f"Output root : {args.output_dir}")
    print(f"Log dir     : {log_dir}")
    print()

    rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    success_count = 0

    batch_start = time.time()
    total_runs = len(selected_prompts)
    for local_idx, prompt in enumerate(selected_prompts, start=1):
        global_idx = args.start_index + local_idx - 1
        prompt_slug = _slugify(prompt)
        run_name = f"run_{global_idx:04d}_{prompt_slug}"
        run_output_dir = args.output_dir / run_name

        cmd = _build_sd_cmd(args, prompt, run_output_dir)

        print(_c(f"[{local_idx:03d}/{total_runs:03d}]", _Style.BOLD), _truncate(prompt, 100))
        print(_c("  output:", _Style.DIM), run_output_dir)

        if args.dry_run:
            print(_c("  dry-run command:", _Style.DIM), " ".join(cmd))
            rows.append({
                "index": global_idx,
                "status": "DRY",
                "elapsed": 0.0,
                "prompt": prompt,
            })
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "mean": "",
                "max": "",
                "status": "DRY",
            })
            continue

        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        stdout_path = log_dir / f"{run_name}.stdout.log"
        stderr_path = log_dir / f"{run_name}.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")

        eval_stats = _extract_eval_stats(proc.stdout or "")
        if proc.returncode == 0:
            success_count += 1
            status = _c("OK", _Style.GREEN, _Style.BOLD)
            rows.append({
                "index": global_idx,
                "status": "OK",
                "elapsed": elapsed,
                "prompt": prompt,
            })
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "mean": eval_stats["mean"],
                "max": eval_stats["max"],
                "status": "OK",
            })
            print(_c(f"  status: {status}  time: {elapsed:.2f}s", _Style.DIM))
        else:
            status = _c("FAIL", _Style.RED, _Style.BOLD)
            rows.append({
                "index": global_idx,
                "status": "FAIL",
                "elapsed": elapsed,
                "prompt": prompt,
            })
            eval_rows.append({
                "index": global_idx,
                "prompt": prompt,
                "mean": eval_stats["mean"],
                "max": eval_stats["max"],
                "status": "FAIL",
            })
            print(_c(f"  status: {status}  time: {elapsed:.2f}s  code: {proc.returncode}", _Style.DIM))
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

    csv_path = args.output_dir / "batch_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["index", "prompt", "mean", "max", "status"])
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(row)

    print()
    _title("Result")
    print(f"Success: {success_count}/{len(rows)}")
    print(f"Elapsed: {total_elapsed:.2f}s")
    print(f"Summary CSV: {csv_path}")
    print(f"Batch logs : {log_dir}")

    if success_count != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

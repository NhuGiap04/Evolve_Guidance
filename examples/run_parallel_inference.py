#!/usr/bin/env python3
"""Launch DAS parallel inference with accelerate."""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch DAS.py with accelerate for multi-GPU data-parallel inference.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sd.py:pick",
        help="Config target passed to DAS.py, e.g. config/sd.py:pick or config/sdxl.py:pick.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help="Number of accelerate worker processes.",
    )
    parser.add_argument(
        "--accelerate-bin",
        type=str,
        default="accelerate",
        help="Accelerate executable name or path.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated GPU ids for CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.num_processes < 1:
        raise ValueError("--num-processes must be >= 1")

    cmd = [
        args.accelerate_bin,
        "launch",
        "--num_processes",
        str(args.num_processes),
        "DAS.py",
        "--config",
        args.config,
    ]

    env = os.environ.copy()
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print("Command:")
    print(" ".join(cmd))
    if args.gpu_ids:
        print(f"CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

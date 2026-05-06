#!/usr/bin/env python3
"""Batch runner for runs/single/approximate_sdxl.py using prompts from .txt or .json."""

import sys

from runs.gradient_sdxl_batch import main


def _ensure_arg(flag: str, value: str) -> None:
    if flag in sys.argv:
        return
    sys.argv.extend([flag, value])


def _has_any_flag(flags) -> bool:
    return any(flag in sys.argv for flag in flags)


def _main() -> int:
    if not _has_any_flag(["--sdxl-script", "--gradient-script"]):
        _ensure_arg("--sdxl-script", "runs/single/approximate_sdxl.py")
    if "--output-dir" not in sys.argv:
        _ensure_arg("--output-dir", "logs/sdxl_batch_approx")
    return main()


if __name__ == "__main__":
    raise SystemExit(_main())

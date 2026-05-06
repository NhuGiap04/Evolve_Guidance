#!/usr/bin/env python3
"""Compatibility shim for older invocations."""

from runs.approximate_sd_batch import _main


if __name__ == "__main__":
    raise SystemExit(_main())

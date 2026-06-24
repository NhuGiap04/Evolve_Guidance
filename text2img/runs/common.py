import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO


BASE_SAMPLE_FIELDS = (
    "num_steps",
    "batch_size",
    "guidance_scale",
    "eta",
    "num_particles",
    "batch_p",
    "stein_step",
    "stein_loop",
    "stein_kernel",
    "stein_repulsion",
    "stein_adagrad_eps",
    "kl_coeff",
)

APPROX_SAMPLE_FIELDS = (
    "soft_temperature",
    "prediction_model",
    "predicted_samples",
    "lookahead_steps",
)

BASE_SINGLE_ARG_FIELDS = (
    "config",
    "negative_prompt",
    "device",
    "seed",
    *BASE_SAMPLE_FIELDS,
    "start",
    "end",
)

APPROX_SINGLE_ARG_FIELDS = (*BASE_SINGLE_ARG_FIELDS, *APPROX_SAMPLE_FIELDS)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def apply_config_overrides(config: Any, args: argparse.Namespace, extra_sample_fields: Iterable[str] = ()) -> Any:
    if args.seed is not None:
        config.seed = args.seed

    for field in (*BASE_SAMPLE_FIELDS, *extra_sample_fields):
        value = getattr(args, field)
        if value is not None:
            setattr(config.sample, field, value)

    config.sample.start = int(args.start)
    config.sample.end = config.sample.num_steps if args.end is None else int(args.end)
    return config


def single_prompt_args(
    args: argparse.Namespace,
    prompt: str,
    output_dir: Path,
    fields: Sequence[str],
) -> argparse.Namespace:
    values = {field: getattr(args, field) for field in fields}
    values.update(prompt=prompt, output_dir=str(output_dir))
    return argparse.Namespace(**values)


def pipeline_config_payload(
    *,
    args: argparse.Namespace,
    config: Any,
    runner: str,
    pipeline_type: str,
    script_path: Path,
) -> Dict[str, Any]:
    return {
        "runner": runner,
        "created_at_unix": float(time.time()),
        "pipeline_type": pipeline_type,
        "pipeline_script": str(script_path),
        "config_name": args.config,
        "model": str(config.pretrained.model),
        "model_revision": str(getattr(config.pretrained, "revision", "")),
        "reward_fn": str(config.reward_fn),
        "seed": int(config.seed),
        "device": args.device,
        "negative_prompt": args.negative_prompt,
        "sample": json_safe(config.sample.to_dict()),
        "batch_args": {k: json_safe(v) for k, v in vars(args).items()},
    }


def build_average_csv_row(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    label: str = "avg",
) -> Optional[Dict[str, str]]:
    averages: Dict[str, str] = {field: "" for field in fieldnames}
    if "index" in averages:
        averages["index"] = label
    elif "prompt" in averages:
        averages["prompt"] = label

    has_average = False
    for field in fieldnames:
        if field in {"index", "prompt", "status"}:
            continue

        values = []
        for row in rows:
            value = row.get(field, "")
            if value in ("", None):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                values.append(numeric_value)

        if values:
            averages[field] = f"{sum(values) / len(values):.6f}"
            has_average = True

    return averages if has_average else None


def _truncate_progress_text(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class DenoisingProgress:
    tensor_inputs: Sequence[str] = ()

    def __init__(
        self,
        total_steps: int,
        *,
        prompt: str = "",
        width: int = 28,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.total_steps = max(1, int(total_steps))
        self.prompt = _truncate_progress_text(prompt, 56)
        self.width = max(10, int(width))
        self.stream = stream or sys.__stdout__
        self.last_len = 0

    def __call__(self, pipe: Any, step_index: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> None:
        del pipe, timestep, callback_kwargs
        current = min(max(int(step_index) + 1, 0), self.total_steps)
        fraction = current / self.total_steps
        filled = min(self.width, int(round(self.width * fraction)))
        bar = "#" * filled + "-" * (self.width - filled)
        prompt_part = f"  prompt: {self.prompt}" if self.prompt else ""
        line = f"  denoising [{bar}] {current:03d}/{self.total_steps:03d} {fraction * 100:5.1f}%{prompt_part}"
        self.stream.write("\r" + line.ljust(self.last_len))
        self.stream.flush()
        self.last_len = len(line)
        if current >= self.total_steps:
            self.stream.write("\n")
            self.stream.flush()

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


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

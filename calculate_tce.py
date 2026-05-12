import argparse
import csv
import json
import os
from pathlib import Path

try:
    from image_diversity import ClipMetrics
except ImportError as exc:
    ClipMetrics = None
    IMAGE_DIVERSITY_IMPORT_ERROR = exc


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def discover_eval_runs(roots, image_glob):
    eval_dirs = []
    for root in roots:
        root = resolve_path(root)
        print(f"Scanning root: {root} exists={root.exists()}")
        if not root.exists():
            continue
        if (root / "final_rewards.json").exists() and list(root.glob(image_glob)):
            eval_dirs.append(root)
            continue
        for metadata_path in root.rglob("final_rewards.json"):
            run_dir = metadata_path.parent
            if list(run_dir.glob(image_glob)):
                eval_dirs.append(run_dir)
    return sorted(set(eval_dirs))


def common_batch_root(eval_dirs):
    if not eval_dirs:
        return Path.cwd()
    common = Path(os.path.commonpath([str(path) for path in eval_dirs]))
    if common.name.startswith("run_"):
        return common.parent
    if common.name.startswith(("pick_seed", "clip_seed", "seg_seed")):
        return common.parent.parent
    return common


def load_metadata(run_dir):
    metadata_path = run_dir / "final_rewards.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def batch_run_name(run_dir):
    if run_dir.parent.name.startswith("run_"):
        return run_dir.parent.name
    return run_dir.name


def summarize(values):
    values = [float(value) for value in values if value is not None and value == value]
    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def calculate_tce(clip_metrics, image_paths, requested_n_eigs, batch_size):
    if len(image_paths) < 2:
        return float("nan"), 0
    effective_n_eigs = min(requested_n_eigs, len(image_paths) - 1)
    if effective_n_eigs < 1:
        return float("nan"), 0
    clip_metrics.n_eigs = effective_n_eigs
    tce = clip_metrics.tce(
        str(image_paths[0].parent),
        img_names=[path.name for path in image_paths],
        batch_size=batch_size,
    )
    return float(tce), effective_n_eigs


def main():
    parser = argparse.ArgumentParser(description="Calculate TCE with fibarrola/image_diversity")
    parser.add_argument("--eval-root", type=Path, required=True, help="Batch root or single run directory")
    parser.add_argument("--image-glob", default="sample_*.png", help="Image glob inside each run directory")
    parser.add_argument("--n-eigs", type=int, default=20, help="Requested TCE eigenvalue count")
    parser.add_argument("--batch-size", type=int, default=16, help="CLIP encoding batch size")
    parser.add_argument("--output-name", default="tce_image_diversity.csv", help="Per-run CSV filename")
    args = parser.parse_args()

    if ClipMetrics is None:
        raise ImportError(
            "image-diversity is required. Install it with: pip install image-diversity"
        ) from IMAGE_DIVERSITY_IMPORT_ERROR

    # image-diversity 0.1.6 has a bug when device is passed explicitly, so use auto-device.
    clip_metrics = ClipMetrics(n_eigs=args.n_eigs)
    eval_dirs = discover_eval_runs([args.eval_root], args.image_glob)
    print(f"Found {len(eval_dirs)} eval runs")

    rows = []
    for eval_dir in eval_dirs:
        metadata = load_metadata(eval_dir)
        image_files = metadata.get("image_files") or [path.name for path in sorted(eval_dir.glob(args.image_glob))]
        image_paths = [eval_dir / name for name in image_files if (eval_dir / name).exists()]
        tce, effective_n_eigs = calculate_tce(
            clip_metrics,
            image_paths=image_paths,
            requested_n_eigs=args.n_eigs,
            batch_size=args.batch_size,
        )
        row = {
            "run_id": batch_run_name(eval_dir),
            "eval_dir": str(eval_dir),
            "config": metadata.get("config", eval_dir.name),
            "prompt": metadata.get("prompt", ""),
            "num_images": len(image_paths),
            "requested_n_eigs": args.n_eigs,
            "tce_n_eigs": effective_n_eigs,
            "tce": tce,
        }
        rows.append(row)
        with (eval_dir / args.output_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    batch_root = common_batch_root(eval_dirs)
    batch_csv = batch_root / "tce_image_diversity_batch.csv"
    fields = ["run_id", "eval_dir", "config", "prompt", "num_images", "requested_n_eigs", "tce_n_eigs", "tce"]
    with batch_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize([row["tce"] for row in rows])
    summary_csv = batch_root / "tce_image_diversity_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerow({"metric": "tce", **summary})

    print(f"Saved per-run TCE files as {args.output_name}")
    print(f"Saved batch TCE rows: {batch_csv.resolve()}")
    print(f"Saved batch TCE summary: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()

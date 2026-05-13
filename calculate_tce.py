import argparse
import csv
import json
import os
import re
import shutil
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


def load_per_image_scores(run_dir, image_files, score_column, csv_name):
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or score_column not in reader.fieldnames or "image" not in reader.fieldnames:
            return None
        scores_by_image = {}
        for row in reader:
            image_name = row.get("image")
            if not image_name:
                continue
            try:
                scores_by_image[image_name] = float(row[score_column])
            except (TypeError, ValueError):
                continue
    if not scores_by_image:
        return None
    scores = []
    for image_file in image_files:
        if image_file not in scores_by_image:
            return None
        scores.append(scores_by_image[image_file])
    return score_column, scores


def scores_from_metadata(metadata, image_files, best_reward):
    reward_keys = []
    if best_reward in {"auto", "eval"}:
        reward_keys.append("eval_rewards")
    if best_reward in {"auto", "steer"}:
        reward_keys.append("steer_rewards")

    for reward_key in reward_keys:
        scores = metadata.get(reward_key)
        if isinstance(scores, list) and len(scores) == len(image_files):
            return reward_key, [float(score) for score in scores]
    return None


def select_best_image(run_dir, metadata, image_paths, best_reward, best_score_column, per_image_csv_name):
    if not image_paths:
        return None
    image_files = [path.name for path in image_paths]
    score_result = None
    if best_score_column:
        score_result = load_per_image_scores(run_dir, image_files, best_score_column, per_image_csv_name)
    if score_result is None:
        score_result = scores_from_metadata(metadata, image_files, best_reward)
    if score_result is None:
        return None

    score_source, scores = score_result
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    return {
        "image_path": image_paths[best_idx],
        "image": image_paths[best_idx].name,
        "score_source": score_source,
        "score": scores[best_idx],
    }


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


def safe_name(text):
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return name.strip("._") or "run"


def stage_best_images(selected_rows, stage_dir):
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_paths = []
    for idx, row in enumerate(selected_rows):
        source_path = row["image_path"]
        staged_name = f"best_{idx:05d}_{safe_name(row['run_id'])}_{source_path.name}"
        staged_path = stage_dir / staged_name
        shutil.copy2(source_path, staged_path)
        row["staged_image"] = staged_name
        staged_paths.append(staged_path)
    return staged_paths


def main():
    parser = argparse.ArgumentParser(description="Calculate TCE with fibarrola/image_diversity")
    parser.add_argument("--eval-root", type=Path, required=True, help="Batch root or single run directory")
    parser.add_argument("--image-glob", default="sample_*.png", help="Image glob inside each run directory")
    parser.add_argument("--n-eigs", type=int, default=20, help="Requested TCE eigenvalue count")
    parser.add_argument("--batch-size", type=int, default=16, help="CLIP encoding batch size")
    parser.add_argument("--output-name", default="tce_image_diversity.csv", help="Per-run CSV filename")
    parser.add_argument(
        "--best-per-prompt",
        action="store_true",
        help="Pick the best image from each prompt/run, then calculate one TCE over those selected images",
    )
    parser.add_argument(
        "--best-reward",
        choices=["auto", "eval", "steer"],
        default="auto",
        help="Reward list in final_rewards.json used by --best-per-prompt. auto prefers eval_rewards, then steer_rewards.",
    )
    parser.add_argument(
        "--best-score-column",
        default=None,
        help="Optional eval_per_image.csv column used by --best-per-prompt to choose the best image",
    )
    parser.add_argument(
        "--per-image-csv-name",
        default="eval_per_image.csv",
        help="Per-image CSV filename used with --best-score-column",
    )
    parser.add_argument(
        "--best-stage-dir",
        type=Path,
        default=None,
        help="Directory where selected best images are copied before batch TCE",
    )
    args = parser.parse_args()

    if ClipMetrics is None:
        raise ImportError(
            "image-diversity is required. Install it with: pip install image-diversity"
        ) from IMAGE_DIVERSITY_IMPORT_ERROR

    # image-diversity 0.1.6 has a bug when device is passed explicitly, so use auto-device.
    clip_metrics = ClipMetrics(n_eigs=args.n_eigs)
    eval_dirs = discover_eval_runs([args.eval_root], args.image_glob)
    print(f"Found {len(eval_dirs)} eval runs")

    if args.best_per_prompt:
        selected_rows = []
        skipped = 0
        for eval_dir in eval_dirs:
            metadata = load_metadata(eval_dir)
            image_files = metadata.get("image_files") or [path.name for path in sorted(eval_dir.glob(args.image_glob))]
            image_paths = [eval_dir / name for name in image_files if (eval_dir / name).exists()]
            best = select_best_image(
                eval_dir,
                metadata=metadata,
                image_paths=image_paths,
                best_reward=args.best_reward,
                best_score_column=args.best_score_column,
                per_image_csv_name=args.per_image_csv_name,
            )
            if best is None:
                skipped += 1
                print(f"[WARN] Skipping {eval_dir}: no usable reward scores for best-image selection")
                continue
            selected_rows.append({
                "run_id": batch_run_name(eval_dir),
                "eval_dir": str(eval_dir),
                "config": metadata.get("config", eval_dir.name),
                "prompt": metadata.get("prompt", ""),
                **best,
            })

        batch_root = common_batch_root(eval_dirs)
        stage_dir = resolve_path(args.best_stage_dir) if args.best_stage_dir else batch_root / "tce_best_per_prompt_images"
        staged_paths = stage_best_images(selected_rows, stage_dir)
        tce, effective_n_eigs = calculate_tce(
            clip_metrics,
            image_paths=staged_paths,
            requested_n_eigs=args.n_eigs,
            batch_size=args.batch_size,
        )

        selection_csv = batch_root / "tce_best_per_prompt_selection.csv"
        selection_fields = [
            "run_id",
            "eval_dir",
            "config",
            "prompt",
            "image",
            "score_source",
            "score",
            "staged_image",
        ]
        with selection_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=selection_fields)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow({field: row[field] for field in selection_fields})

        result_csv = batch_root / "tce_best_per_prompt.csv"
        result_row = {
            "eval_root": str(resolve_path(args.eval_root)),
            "stage_dir": str(stage_dir),
            "num_prompts": len(selected_rows),
            "skipped_prompts": skipped,
            "requested_n_eigs": args.n_eigs,
            "tce_n_eigs": effective_n_eigs,
            "tce": tce,
        }
        with result_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result_row.keys()))
            writer.writeheader()
            writer.writerow(result_row)

        print(f"Selected {len(selected_rows)} best images; skipped {skipped} runs")
        print(f"Saved best-image selection: {selection_csv.resolve()}")
        print(f"Saved best-image TCE: {result_csv.resolve()}")
        return

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

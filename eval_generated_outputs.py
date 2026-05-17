import argparse
import os
import sys
import csv
import json
import tempfile
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import shutil

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms

from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer
try:
    from seg.scorers.aesthetic_scorer import AestheticScorer
except Exception as exc:
    AestheticScorer = None
    AESTHETIC_IMPORT_ERROR = exc
try:
    from seg.scorers.hpsv2_scorer import HPSv2Scorer
except Exception as exc:
    HPSv2Scorer = None
    HPS_IMPORT_ERROR = exc
try:
    import das.rewards as das_rewards
except ImportError:
    das_rewards = None
try:
    from scipy.spatial.distance import pdist
except ImportError:
    pdist = None
try:
    import lpips
except ImportError:
    lpips = None
try:
    from image_diversity import ClipMetrics
except ImportError:
    ClipMetrics = None

def build_reward_scorer(name, dtype, device):
    normalized = name.lower()
    if normalized in {"aesthetic", "aesthetic_score"}:
        if AestheticScorer is not None:
            return AestheticScorer(dtype=dtype, device=device)
        if das_rewards is not None:
            return das_rewards.aesthetic_score(torch_dtype=torch.float32, device=str(device))
        raise ImportError(f"aesthetic scorer unavailable: {AESTHETIC_IMPORT_ERROR}")
    if normalized in {"hps", "hps_score", "hpsv2"}:
        if HPSv2Scorer is not None:
            return HPSv2Scorer(dtype=dtype, device=device)
        if das_rewards is not None:
            return das_rewards.hps_score(inference_dtype=torch.float32, device=str(device))
        raise ImportError(f"hps scorer unavailable: {HPS_IMPORT_ERROR}")
    if normalized in {"clip", "clip_score"}:
        return CLIPScorer(dtype=dtype, device=device)
    if normalized in {"pick", "pick_score"}:
        return PickScoreScorer(dtype=dtype, device=device)
    if normalized in {"image_reward", "imagereward", "image_reward_score"}:
        return ImageRewardScorer(dtype=dtype, device=device)
    raise ValueError(f"Unsupported reward scorer: {name}")

def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    repo_path = (Path(__file__).parent.parent / path).resolve()
    if repo_path.exists():
        return repo_path
    return cwd_path

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

def logged_reward_series(metadata, preferred_reward="eval"):
    reward_order = {
        "eval": [("eval_rewards", "eval_reward_name"), ("steer_rewards", "steer_reward_name")],
        "steer": [("steer_rewards", "steer_reward_name")],
    }
    for reward_key, name_key in reward_order.get(preferred_reward, reward_order["eval"]):
        rewards = metadata.get(reward_key)
        if rewards is not None:
            return rewards, metadata.get(name_key, reward_key.replace("_rewards", ""))
    return None, None

def prompt_for_run(run_dir, metadata, prompt_override):
    if prompt_override is not None:
        return prompt_override
    prompt = metadata.get("prompt")
    if prompt:
        return prompt
    raise ValueError(f"No prompt found for {run_dir}. Set PROMPT_OVERRIDE or keep final_rewards.json next to images.")

def batch_run_name(run_dir):
    if run_dir.parent.name.startswith("run_"):
        return run_dir.parent.name
    return run_dir.name

def collect_best_images_per_prompt(eval_dirs, prompt_override=None, preferred_reward="eval", image_glob="sample_*.png"):
    best_by_prompt = {}
    skipped = []
    for eval_dir in eval_dirs:
        metadata = load_metadata(eval_dir)
        try:
            prompt = prompt_for_run(eval_dir, metadata, prompt_override)
        except ValueError as exc:
            skipped.append({"eval_dir": str(eval_dir), "reason": str(exc)})
            continue

        rewards, reward_name = logged_reward_series(metadata, preferred_reward=preferred_reward)
        if rewards is None:
            skipped.append({"eval_dir": str(eval_dir), "prompt": prompt, "reason": "no logged eval_rewards or steer_rewards"})
            continue

        image_files = metadata.get("image_files") or [path.name for path in sorted(eval_dir.glob(image_glob))]
        candidates = []
        for image_name, reward in zip(image_files, rewards):
            image_path = eval_dir / image_name
            if image_path.exists() and reward is not None:
                candidates.append((float(reward), image_path, image_name))
        if not candidates:
            skipped.append({"eval_dir": str(eval_dir), "prompt": prompt, "reason": "no image/reward pairs found"})
            continue

        reward, image_path, image_name = max(candidates, key=lambda item: item[0])
        row = {
            "prompt": prompt,
            "run_id": batch_run_name(eval_dir),
            "eval_dir": str(eval_dir),
            "config": metadata.get("config", eval_dir.name),
            "image": image_name,
            "image_path": str(image_path),
            "reward_name": reward_name,
            "reward": reward,
        }
        update_best_image_selection(best_by_prompt, row)
    return [best_by_prompt[prompt] for prompt in sorted(best_by_prompt)], skipped

def update_best_image_selection(best_by_prompt, row):
    prompt = row["prompt"]
    previous = best_by_prompt.get(prompt)
    if previous is None or row["reward"] > previous["reward"]:
        best_by_prompt[prompt] = row

def normalize_reward_name(name):
    normalized = (name or "").lower()
    aliases = {
        "aesthetic_score": "aesthetic",
        "hps_score": "hps",
        "hpsv2": "hps",
        "image_reward_score": "image_reward",
        "imagereward": "image_reward",
        "pick_score": "pick",
        "clip_score": "clip",
    }
    return aliases.get(normalized, normalized)

def tce_fallback_reward_for_run(metadata, requested_fallback):
    if requested_fallback != "steer":
        return requested_fallback
    reward_name = normalize_reward_name(metadata.get("steer_reward_name"))
    if reward_name in {"aesthetic", "hps", "image_reward", "pick", "clip"}:
        return reward_name
    return None

def load_image_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)

def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

def score_reward(scorer, reward_name, images, prompts):
    if reward_name.lower() in {"aesthetic", "aesthetic_score"}:
        try:
            return scorer(images)
        except TypeError:
            return scorer(images, prompts)
    return scorer(images, prompts)

def load_diversity_models(device, run_diversity):
    if not run_diversity:
        return None, None, None
    if pdist is None:
        raise ImportError("CLIP diversity requires scipy. Install scipy or set RUN_DIVERSITY=False.")
    if lpips is None:
        raise ImportError("LPIPS diversity requires lpips. Install lpips or set RUN_DIVERSITY=False.")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    return clip_model, clip_processor, lpips_model

def calculate_image_diversity_tce(image_paths, device, n_eigs=20, batch_size=16):
    if len(image_paths) < 2:
        return {"num_images": len(image_paths), "tce": np.nan}
    if ClipMetrics is None:
        raise ImportError(
            "image-diversity is required for best-image TCE. Install it with: pip install image-diversity"
        )
    with tempfile.TemporaryDirectory(prefix="best_image_tce_") as temp_dir:
        temp_dir = Path(temp_dir)
        for idx, image_path in enumerate(image_paths):
            suffix = image_path.suffix or ".png"
            shutil.copy2(image_path, temp_dir / f"selected_{idx:04d}{suffix}")
        kwargs = {"n_eigs": n_eigs, "batch_size": batch_size}
        if device is not None:
            kwargs["device"] = str(device)
        try:
            clip_metrics = ClipMetrics(**kwargs)
        except TypeError:
            clip_metrics = ClipMetrics()
        try:
            tce = clip_metrics.tce(str(temp_dir), n_eigs=n_eigs, batch_size=batch_size)
        except TypeError:
            tce = clip_metrics.tce(str(temp_dir))
    return {"num_images": len(image_paths), "tce": float(tce)}

def load_lpips_tensor(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)

def calculate_diversity_metrics(image_paths, clip_model, clip_processor, lpips_model, device, k=20):
    if len(image_paths) < 2:
        return {
            "num_images": len(image_paths),
            "clip_pairwise_mean": np.nan,
            "clip_pairwise_std": np.nan,
            "clip_pairwise_min": np.nan,
            "clip_pairwise_max": np.nan,
            "clip_pairwise_std_error": np.nan,
            "tce": np.nan,
            "lpips_mean": np.nan,
            "lpips_std": np.nan,
            "lpips_min": np.nan,
            "lpips_max": np.nan,
        }
    embeddings = []
    lpips_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs).detach().float().cpu().numpy().squeeze()
        embeddings.append(embedding)
        lpips_images.append(load_lpips_tensor(image_path).to(device))
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pairwise_distances = pdist(embeddings, metric="cosine")
    clip_pairwise = summarize(pairwise_distances)
    clip_std_error = float(np.std(pairwise_distances, ddof=0) / np.sqrt(pairwise_distances.size))
    covariance_matrix = np.cov(embeddings, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)[-min(k, covariance_matrix.shape[0]):]
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    tce = float((len(eigenvalues) / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(eigenvalues)))
    lpips_distances = []
    for i, j in combinations(range(len(lpips_images)), 2):
        with torch.no_grad():
            lpips_distances.append(float(lpips_model(lpips_images[i], lpips_images[j]).item()))
    lpips_stats = summarize(lpips_distances)
    return {
        "num_images": len(image_paths),
        "clip_pairwise_mean": clip_pairwise["mean"],
        "clip_pairwise_std": clip_pairwise["std"],
        "clip_pairwise_min": clip_pairwise["min"],
        "clip_pairwise_max": clip_pairwise["max"],
        "clip_pairwise_std_error": clip_std_error,
        "tce": tce,
        "lpips_mean": lpips_stats["mean"],
        "lpips_std": lpips_stats["std"],
        "lpips_min": lpips_stats["min"],
        "lpips_max": lpips_stats["max"],
    }

def main():
    parser = argparse.ArgumentParser(description="Offline Batch Evaluation Script")
    parser.add_argument('--eval-root', type=str, required=True, help='Path to batch root or run folder to evaluate')
    parser.add_argument('--prompt-override', type=str, default=None, help='Override prompt if final_rewards.json is missing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--run-diversity', action='store_true', help='Compute diversity metrics')
    parser.add_argument('--tce-k', type=int, default=20, help='TCE K value')
    parser.add_argument(
        '--tce-reward-source',
        type=str,
        default='steer',
        choices=['eval', 'steer'],
        help='Logged reward source to prefer when selecting the best image per prompt for image-diversity TCE',
    )
    parser.add_argument(
        '--tce-fallback-reward',
        type=str,
        default='steer',
        choices=['steer', 'aesthetic', 'hps', 'image_reward', 'pick', 'clip'],
        help='Offline evaluator reward used for best-image TCE when a run has no logged final rewards; steer uses the run inference reward',
    )
    args = parser.parse_args()

    EVAL_ROOTS = [Path(args.eval_root)]
    PROMPT_OVERRIDE = args.prompt_override
    REWARD_NAMES = ["aesthetic", "hps", "image_reward", "pick", "clip"]
    RUN_DIVERSITY = args.run_diversity
    TCE_K = args.tce_k
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    IMAGE_GLOB = "sample_*.png"
    PER_IMAGE_CSV_NAME = "eval_per_image.csv"
    RUN_SUMMARY_CSV_NAME = "eval_run_summary.csv"
    RUN_DIVERSITY_CSV_NAME = "eval_run_diversity_summary.csv"
    BATCH_SUMMARY_CSV_NAME = "eval_batch_summary_from_run_stats.csv"
    BATCH_DIVERSITY_CSV_NAME = "eval_batch_diversity_summary_from_run_stats.csv"
    BEST_IMAGE_MANIFEST_CSV_NAME = "eval_best_images_for_tce.csv"
    BEST_IMAGE_DIVERSITY_CSV_NAME = "eval_best_image_diversity_summary.csv"

    eval_dirs = discover_eval_runs(EVAL_ROOTS, IMAGE_GLOB)
    print(f"Found {len(eval_dirs)} eval runs")
    for run_dir in eval_dirs[:10]:
        print(run_dir)

    device = torch.device(DEVICE if DEVICE.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device={device}, dtype={dtype}")

    scorers = {name: build_reward_scorer(name, dtype=dtype, device=device) for name in REWARD_NAMES}
    clip_diversity_model, clip_diversity_processor, lpips_model = load_diversity_models(device, RUN_DIVERSITY)

    run_summary_rows = []
    run_diversity_rows = []
    offline_best_by_prompt = {}

    for eval_dir in eval_dirs:
        metadata = load_metadata(eval_dir)
        prompt = prompt_for_run(eval_dir, metadata, PROMPT_OVERRIDE)
        run_id = batch_run_name(eval_dir)
        config = metadata.get("config", eval_dir.name)
        tce_fallback_reward = tce_fallback_reward_for_run(metadata, args.tce_fallback_reward)
        image_files = metadata.get("image_files") or [path.name for path in sorted(eval_dir.glob(IMAGE_GLOB))]
        image_paths = [eval_dir / name for name in image_files if (eval_dir / name).exists()]
        per_run_image_rows = []
        per_run_summary_rows = []
        scores_by_reward = defaultdict(list)

        for offset in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[offset : offset + BATCH_SIZE]
            images = torch.stack([load_image_tensor(path) for path in batch_paths]).to(device=device)
            prompts = [prompt] * len(batch_paths)
            with torch.no_grad():
                batch_scores_by_reward = {}
                for reward_name, scorer in scorers.items():
                    scores = score_reward(scorer, reward_name, images, prompts).detach().float().cpu().tolist()
                    batch_scores_by_reward[reward_name] = scores
                    scores_by_reward[reward_name].extend(scores)
            for local_idx, image_path in enumerate(batch_paths):
                row = {
                    "run_id": run_id,
                    "eval_dir": str(eval_dir),
                    "config": config,
                    "image": image_path.name,
                    "prompt": prompt,
                }
                for reward_name in REWARD_NAMES:
                    row[reward_name] = batch_scores_by_reward[reward_name][local_idx]
                per_run_image_rows.append(row)
                if tce_fallback_reward is not None:
                    update_best_image_selection(
                        offline_best_by_prompt,
                        {
                            "prompt": prompt,
                            "run_id": run_id,
                            "eval_dir": str(eval_dir),
                            "config": config,
                            "image": image_path.name,
                            "image_path": str(image_path),
                            "reward_name": tce_fallback_reward,
                            "reward": float(row[tce_fallback_reward]),
                        },
                    )
        for reward_name in REWARD_NAMES:
            stats = summarize(scores_by_reward[reward_name])
            per_run_summary_rows.append({
                "run_id": run_id,
                "eval_dir": str(eval_dir),
                "config": config,
                "prompt": prompt,
                "reward": reward_name,
                **stats,
            })
        run_summary_rows.extend(per_run_summary_rows)
        if RUN_DIVERSITY:
            diversity_stats = calculate_diversity_metrics(
                image_paths,
                clip_model=clip_diversity_model,
                clip_processor=clip_diversity_processor,
                lpips_model=lpips_model,
                device=device,
                k=TCE_K,
            )
            per_run_diversity_row = {
                "run_id": run_id,
                "eval_dir": str(eval_dir),
                "config": config,
                "prompt": prompt,
                **diversity_stats,
            }
            run_diversity_rows.append(per_run_diversity_row)
        per_image_fields = ["run_id", "eval_dir", "config", "image", "prompt", *REWARD_NAMES]
        with (eval_dir / PER_IMAGE_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_fields)
            writer.writeheader()
            writer.writerows(per_run_image_rows)
        run_summary_fields = ["run_id", "eval_dir", "config", "prompt", "reward", "count", "mean", "std", "min", "max"]
        with (eval_dir / RUN_SUMMARY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_summary_fields)
            writer.writeheader()
            writer.writerows(per_run_summary_rows)
        if RUN_DIVERSITY:
            run_diversity_fields = [
                "run_id",
                "eval_dir",
                "config",
                "prompt",
                "num_images",
                "clip_pairwise_mean",
                "clip_pairwise_std",
                "clip_pairwise_min",
                "clip_pairwise_max",
                "clip_pairwise_std_error",
                "tce",
                "lpips_mean",
                "lpips_std",
                "lpips_min",
                "lpips_max",
            ]
            with (eval_dir / RUN_DIVERSITY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=run_diversity_fields)
                writer.writeheader()
                writer.writerow(per_run_diversity_row)
    print(f"Scored {sum(row['count'] for row in run_summary_rows if row['reward'] == REWARD_NAMES[0])} images across {len(eval_dirs)} runs")
    if RUN_DIVERSITY:
        print(f"Computed diversity metrics for {len(run_diversity_rows)} runs")

    batch_summary_rows = []
    batch_diversity_rows = []
    run_stat_names = ["mean", "std", "min", "max"]
    for reward_name in REWARD_NAMES:
        reward_rows = [row for row in run_summary_rows if row["reward"] == reward_name]
        for run_stat in run_stat_names:
            values = [row[run_stat] for row in reward_rows]
            stats = summarize(values)
            batch_summary_rows.append({
                "reward": reward_name,
                "run_stat": run_stat,
                "num_runs": stats.pop("count"),
                **stats,
            })
    if RUN_DIVERSITY:
        diversity_stat_names = [
            "clip_pairwise_mean",
            "clip_pairwise_std",
            "clip_pairwise_min",
            "clip_pairwise_max",
            "clip_pairwise_std_error",
            "tce",
            "lpips_mean",
            "lpips_std",
            "lpips_min",
            "lpips_max",
        ]
        for diversity_stat in diversity_stat_names:
            values = [row[diversity_stat] for row in run_diversity_rows]
            stats = summarize(values)
            batch_diversity_rows.append({
                "run_stat": diversity_stat,
                "num_runs": stats.pop("count"),
                **stats,
            })
    batch_root = common_batch_root(eval_dirs)
    print(f"Writing batch summaries to: {batch_root}")
    if RUN_DIVERSITY:
        best_image_rows, skipped_best_image_rows = collect_best_images_per_prompt(
            eval_dirs,
            prompt_override=PROMPT_OVERRIDE,
            preferred_reward=args.tce_reward_source,
            image_glob=IMAGE_GLOB,
        )
        logged_best_prompts = {row["prompt"] for row in best_image_rows}
        for prompt, row in offline_best_by_prompt.items():
            if prompt not in logged_best_prompts:
                fallback_row = dict(row)
                fallback_row["reward_name"] = f"offline_{fallback_row['reward_name']}"
                best_image_rows.append(fallback_row)
        best_image_rows = sorted(best_image_rows, key=lambda row: row["prompt"])
        best_manifest_fields = ["prompt", "run_id", "eval_dir", "config", "image", "image_path", "reward_name", "reward"]
        with (batch_root / BEST_IMAGE_MANIFEST_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best_manifest_fields)
            writer.writeheader()
            writer.writerows(best_image_rows)
        best_image_paths = [Path(row["image_path"]) for row in best_image_rows]
        best_tce_stats = calculate_image_diversity_tce(
            best_image_paths,
            device=device,
            n_eigs=TCE_K,
            batch_size=BATCH_SIZE,
        )
        best_diversity_row = {
            "selection": "best_logged_reward_per_prompt",
            "preferred_reward_source": args.tce_reward_source,
            "fallback_reward": args.tce_fallback_reward,
            "num_selected_prompts": len(best_image_rows),
            "num_skipped_runs": len(skipped_best_image_rows),
            **best_tce_stats,
        }
        best_diversity_fields = [
            "selection",
            "preferred_reward_source",
            "fallback_reward",
            "num_selected_prompts",
            "num_skipped_runs",
            "num_images",
            "tce",
        ]
        with (batch_root / BEST_IMAGE_DIVERSITY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best_diversity_fields)
            writer.writeheader()
            writer.writerow(best_diversity_row)
        if skipped_best_image_rows:
            print(f"Skipped {len(skipped_best_image_rows)} runs without usable logged rewards for best-image TCE")
    batch_summary_fields = ["reward", "run_stat", "num_runs", "mean", "std", "min", "max"]
    with (batch_root / BATCH_SUMMARY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_summary_fields)
        writer.writeheader()
        writer.writerows(batch_summary_rows)
    if RUN_DIVERSITY:
        batch_diversity_fields = ["run_stat", "num_runs", "mean", "std", "min", "max"]
        with (batch_root / BATCH_DIVERSITY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_diversity_fields)
            writer.writeheader()
            writer.writerows(batch_diversity_rows)
    print(f"Saved per-run files inside each eval_dir: {PER_IMAGE_CSV_NAME}, {RUN_SUMMARY_CSV_NAME}")
    print(f"Saved batch reward summary from run stats: {(batch_root / BATCH_SUMMARY_CSV_NAME).resolve()}")
    if RUN_DIVERSITY:
        print(f"Saved per-run diversity file inside each eval_dir: {RUN_DIVERSITY_CSV_NAME}")
        print(f"Saved batch diversity summary from run stats: {(batch_root / BATCH_DIVERSITY_CSV_NAME).resolve()}")
        print(f"Saved best-image TCE manifest: {(batch_root / BEST_IMAGE_MANIFEST_CSV_NAME).resolve()}")
        print(f"Saved best-image image-diversity TCE: {(batch_root / BEST_IMAGE_DIVERSITY_CSV_NAME).resolve()}")

if __name__ == "__main__":
    main()

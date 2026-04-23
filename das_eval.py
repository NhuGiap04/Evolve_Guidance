import argparse
import csv
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import pdist
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.aesthetic_scorer import AestheticScorer
from seg.scorers.clip_scorer import CLIPScorer
from seg.scorers.hpsv2_scorer import HPSv2Scorer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple DAS-style folder evaluation for reward and diversity metrics."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Run directory containing eval_vis/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run scorers on.",
    )
    parser.add_argument(
        "--tce-k",
        type=int,
        default=20,
        help="Top-K eigenvalues used for truncated CLIP entropy.",
    )
    return parser.parse_args()


def list_image_paths(eval_vis_dir: Path):
    image_paths = []
    for path in sorted(eval_vis_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lower_name = path.name.lower()
        if "ess" in lower_name or "intermediate_rewards" in lower_name:
            continue
        image_paths.append(path)

    if not image_paths:
        raise ValueError(f"No images found in {eval_vis_dir}")
    return image_paths


def prompt_from_filename(path: Path):
    stem = path.stem
    if "|" in stem:
        left = stem.split("|", 1)[0].strip()
        prompt = left.split("_")[-1]
        return prompt[:-1] if prompt.endswith(" ") else prompt
    return stem


def load_image_tensor(path: Path, device):
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def preprocess_lpips(path: Path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def format_values(values):
    return [f"{float(v):.5f}" for v in values]


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    dtype = torch.float32
    run_dir = Path(args.run_dir)
    eval_vis_dir = run_dir / "eval_vis"
    image_paths = list_image_paths(eval_vis_dir)

    aesthetic_fn = AestheticScorer(dtype=dtype, device=device)
    hps_fn = HPSv2Scorer(dtype=dtype, device=device)
    image_reward_fn = ImageRewardScorer(dtype=dtype, device=device)
    pick_fn = PickScoreScorer(dtype=dtype, device=device)
    clip_fn = CLIPScorer(dtype=dtype, device=device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    reward_scores = {
        "Aesthetic score": [],
        "HPS score": [],
        "Image reward score": [],
        "Pick score": [],
        "CLIP score": [],
    }
    clip_embeddings = []
    lpips_images = []

    for image_path in image_paths:
        prompt = prompt_from_filename(image_path)
        image_tensor = load_image_tensor(image_path, device)

        with torch.no_grad():
            reward_scores["CLIP score"].append(clip_fn(image_tensor, [prompt]).item())
            reward_scores["Aesthetic score"].append(aesthetic_fn(image_tensor).item())
            reward_scores["HPS score"].append(hps_fn(image_tensor, [prompt]).item())
            reward_scores["Image reward score"].append(image_reward_fn(image_tensor, [prompt]).item())
            reward_scores["Pick score"].append(pick_fn(image_tensor, [prompt]).item())

            clip_inputs = clip_processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
            pixel_values = clip_inputs["pixel_values"].to(device)
            embedding = clip_model.get_image_features(pixel_values=pixel_values).detach().cpu().numpy().squeeze()
            clip_embeddings.append(embedding)

        lpips_images.append(preprocess_lpips(image_path).to(device))

    reward_header = []
    reward_values = []
    for metric_name, values in reward_scores.items():
        reward_header.extend([metric_name, f"{metric_name} std"])
        reward_values.extend([np.mean(values), np.std(values)])

    reward_csv_path = run_dir / "eval_results.csv"
    with reward_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(reward_header)
        writer.writerow(format_values(reward_values))

    embeddings = np.asarray(clip_embeddings)
    pairwise_distances = pdist(embeddings, metric="cosine")
    mean_distance = float(np.mean(pairwise_distances))
    std_error = float(np.std(pairwise_distances) / np.sqrt(pairwise_distances.size))

    covariance_matrix = np.cov(embeddings, rowvar=False)
    k = min(args.tce_k, covariance_matrix.shape[0])
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)[-k:]
    eigenvalues = np.clip(eigenvalues, a_min=1e-12, a_max=None)
    tce_k = float((k / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(eigenvalues)))

    lpips_distances = []
    for i in range(len(lpips_images)):
        for j in range(i + 1, len(lpips_images)):
            with torch.no_grad():
                lpips_distances.append(lpips_model(lpips_images[i], lpips_images[j]).item())

    mean_lpips = float(np.mean(lpips_distances))
    std_lpips = float(np.std(lpips_distances))

    diversity_header = [
        "Mean Pairwise Distance (CLIP)",
        "Standard Error of the Distance (CLIP)",
        "Truncated CLIP Entropy (TCE)",
        "Mean LPIPS Distance",
        "Std Dev LPIPS Distance",
    ]
    diversity_values = [mean_distance, std_error, tce_k, mean_lpips, std_lpips]

    diversity_csv_path = run_dir / "eval_diversity_results.csv"
    with diversity_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(diversity_header)
        writer.writerow(format_values(diversity_values))

    print(f"Finished evaluating images in {run_dir}")
    for metric_name, values in reward_scores.items():
        print(f"{metric_name}: {np.mean(values):.5f}")
        print(f"{metric_name} std: {np.std(values):.5f}")
    print(f"Mean Pairwise Distance (CLIP-based Diversity Metric): {mean_distance:.5f}")
    print(f"Standard Error of the Distance: {std_error:.5f}")
    print(f"Truncated CLIP Entropy (TCE): {tce_k:.5f}")
    print(f"Mean LPIPS Distance: {mean_lpips:.5f}")
    print(f"Standard Deviation of LPIPS Distance: {std_lpips:.5f}")
    print(f"Saved reward results to: {reward_csv_path}")
    print(f"Saved diversity results to: {diversity_csv_path}")


if __name__ == "__main__":
    main()

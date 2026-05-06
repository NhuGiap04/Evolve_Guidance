import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    repo_path = (Path(__file__).parent / path).resolve()
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


def load_metadata(run_dir):
    metadata_path = run_dir / "final_rewards.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prompt_for_run(run_dir, metadata, prompt_override):
    if prompt_override is not None:
        return prompt_override
    prompt = metadata.get("prompt")
    if prompt:
        return prompt
    raise ValueError(f"No prompt found for {run_dir}. Set --prompt-override or keep final_rewards.json next to images.")


def common_batch_root(eval_dirs):
    if not eval_dirs:
        return Path.cwd()
    common = Path(os.path.commonpath([str(path) for path in eval_dirs]))
    if common.name.startswith("run_"):
        return common.parent
    if common.name.startswith(("pick_seed", "clip_seed", "seg_seed")):
        return common.parent.parent
    return common


def load_geneval_metadata(metadata_path):
    metadata_path = resolve_path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Geneval metadata file not found: {metadata_path}")
    entries = []
    text = metadata_path.read_text(encoding="utf-8").strip()
    if not text:
        return entries
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Geneval metadata JSON must be a list of objects.")
        return data
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def build_prompt_index(metadata_entries):
    prompt_index = {}
    for idx, entry in enumerate(metadata_entries):
        prompt = entry.get("prompt")
        if not prompt:
            continue
        prompt_index.setdefault(prompt, []).append((idx, entry))
    return prompt_index


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_geneval_imagedir(
    eval_dirs,
    prompt_index,
    image_glob,
    samples_per_prompt,
    prompt_override,
    geneval_imagedir,
):
    matched = 0
    missing = 0
    for eval_dir in eval_dirs:
        metadata = load_metadata(eval_dir)
        prompt = prompt_for_run(eval_dir, metadata, prompt_override)
        if prompt not in prompt_index or not prompt_index[prompt]:
            print(f"[WARN] Prompt not found in geneval metadata: {prompt}")
            missing += 1
            continue
        entry_idx, entry = prompt_index[prompt].pop(0)
        folder_name = f"{entry_idx:05d}"
        prompt_dir = geneval_imagedir / folder_name
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = prompt_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f)

        image_paths = sorted(eval_dir.glob(image_glob))[:samples_per_prompt]
        if len(image_paths) < samples_per_prompt:
            print(f"[WARN] Only {len(image_paths)} images found for {eval_dir}")
        for idx, image_path in enumerate(image_paths):
            target_name = f"{idx:04d}.png"
            link_or_copy(image_path, samples_dir / target_name)

        matched += 1
    return matched, missing


def run_geneval(geneval_repo, imagedir, model_path, outdir, options):
    geneval_repo = resolve_path(geneval_repo)
    eval_script = geneval_repo / "evaluation" / "evaluate_images.py"
    summary_script = geneval_repo / "evaluation" / "summary_scores.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Geneval evaluate_images.py not found at {eval_script}")
    outdir.mkdir(parents=True, exist_ok=True)
    results_file = outdir / "results.jsonl"

    command = [
        sys.executable,
        str(eval_script),
        str(imagedir),
        "--outfile",
        str(results_file),
        "--model-path",
        str(resolve_path(model_path)),
    ]
    for option in options or []:
        command.extend(["--options", option])

    print("[INFO] Running Geneval evaluation...")
    subprocess.run(command, check=True)

    print("[INFO] Running Geneval summary...")
    subprocess.run([sys.executable, str(summary_script), str(results_file)], check=True)

    return results_file


def main():
    parser = argparse.ArgumentParser(description="Run Geneval metrics on batch outputs")
    parser.add_argument("--eval-root", type=str, required=True, help="Path to batch root or run folder to evaluate")
    parser.add_argument("--geneval-prompts", type=str, required=True, help="Path to geneval evaluation_metadata.jsonl")
    parser.add_argument("--geneval-repo", type=str, required=True, help="Path to cloned geneval repository")
    parser.add_argument("--geneval-model-path", type=str, required=True, help="Path to downloaded geneval object detector weights")
    parser.add_argument("--geneval-imagedir", type=str, default=None, help="Where to build geneval-style image folders")
    parser.add_argument("--outdir", type=str, default=None, help="Where to write Geneval results")
    parser.add_argument("--image-glob", type=str, default="sample_*.png", help="Glob for generated images")
    parser.add_argument("--samples-per-prompt", type=int, default=4, help="Number of images per prompt")
    parser.add_argument("--prompt-override", type=str, default=None, help="Override prompt if final_rewards.json is missing")
    parser.add_argument("--geneval-option", action="append", default=[], help="Extra Geneval options, e.g. threshold=0.3")
    args = parser.parse_args()

    eval_dirs = discover_eval_runs([Path(args.eval_root)], args.image_glob)
    if not eval_dirs:
        raise ValueError("No eval runs found. Check --eval-root and --image-glob.")

    metadata_entries = load_geneval_metadata(args.geneval_prompts)
    prompt_index = build_prompt_index(metadata_entries)
    if not prompt_index:
        raise ValueError("No prompts found in geneval metadata file.")

    batch_root = common_batch_root(eval_dirs)
    geneval_imagedir = resolve_path(args.geneval_imagedir) if args.geneval_imagedir else batch_root / "geneval_imagedir"
    outdir = resolve_path(args.outdir) if args.outdir else batch_root / "geneval_results"

    geneval_imagedir.mkdir(parents=True, exist_ok=True)
    matched, missing = build_geneval_imagedir(
        eval_dirs=eval_dirs,
        prompt_index=prompt_index,
        image_glob=args.image_glob,
        samples_per_prompt=args.samples_per_prompt,
        prompt_override=args.prompt_override,
        geneval_imagedir=geneval_imagedir,
    )

    print(f"[INFO] Prepared geneval image dir: {geneval_imagedir}")
    print(f"[INFO] Matched prompts: {matched}, missing: {missing}")

    results_file = run_geneval(
        geneval_repo=args.geneval_repo,
        imagedir=geneval_imagedir,
        model_path=args.geneval_model_path,
        outdir=outdir,
        options=args.geneval_option,
    )

    print(f"[INFO] Geneval results saved: {results_file}")


if __name__ == "__main__":
    main()

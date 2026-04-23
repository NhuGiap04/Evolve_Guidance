# Self-Evolve Guidance

Stein-guided SDXL sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
pip install "numpy<2" lpips
```

## Quick Start

Run one prompt:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --num-steps 80 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.02 \
  --steer-start 20 \
  --steer-end 60 \
  --output-dir logs/sdxl
```

## Save Intermediate Images

Add these flags to also save decoded images from denoising steps:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --num-steps 80 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.02 \
  --steer-start 20 \
  --steer-end 60 \
  --save-intermediate-images \
  --trace-decode-batch-size 1 \
  --intermediate-max-samples 2 \
  --output-dir logs/sdxl
```

## Batch Run Multiple Prompts

Use `examples/run_sdxl_batch.py` to run `examples/sdxl.py` across prompts from a `.txt` or `.json` file.

Run from `.txt` (one prompt per line):

```bash
python examples/run_sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --eval-reward image_reward \
  --device cuda \
  --num-steps 80 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.02 \
  --steer-start 20 \
  --steer-end 60 \
  --output-dir logs/sdxl_batch
```

Run from `.json`:

```bash
python examples/run_sdxl_batch.py \
  --prompts-file prompts/benchmark_ir.json \
  --config seg \
  --eval-reward image_reward \
  --device cuda \
  --output-dir logs/sdxl_batch_json
```

Useful flags:

- `--start-index 100 --max-prompts 50`: run a slice of prompts.
- `--stop-on-error`: stop on first failed prompt.
- `--dry-run`: print commands without running them.
- `--save-intermediate-images --trace-decode-batch-size 1`: save step images for each prompt.

LPIPS evaluation against reference images:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward lpips \
  --lpips-ref-dir path/to/reference_images \
  --lpips-net alex \
  --output-dir logs/sdxl_lpips
```

For LPIPS mode, place at least `batch_size` reference images in `--lpips-ref-dir`.
Files are matched in sorted filename order and resized to the generated image size.

Batch outputs:

- One run directory per prompt under `--output-dir`.
- Per-run logs in `<output-dir>/_batch_logs` (`*.stdout.log`, `*.stderr.log`).

## Evaluate A Saved DAS Run

To evaluate a saved run directory the same way as `das_eval.ipynb`, use:

```bash
python das_eval.py logs/SMC/aesthetic/2024.09.26_01.12.19
```

This reads images from `<run-dir>/eval_vis` and writes:

- `eval_results.csv`
- `eval_diversity_results.csv`

## Multi-GPU Inference With Accelerate

For DAS-style multi-GPU inference, launch one process per GPU with `accelerate`. Each process now receives a distinct prompt slice and rank 0 gathers the results for logging and image saving.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 DAS.py --config config/sdxl.py:pick
```

Notes:

- `config.sample.batch_size` is per GPU.
- Total generated samples per evaluation run are `batch_size * num_processes * max_vis_images`.
- This is data-parallel inference: each GPU loads a full pipeline replica and works on different prompts.

## Main Options

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--prompt`: text prompt
- `--num-steps`: denoising steps
- `--num-particles`: particle count for Stein guidance
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein step size
- `--steer-start`, `--steer-end`: steering window (0-based step index)

## Outputs

Saved in `logs/sdxl/<config>_seed<seed>`:

- Final images (`sample_*.png`)
- Reward traces (`steer_trace.csv`)
- Reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`)

# Self-Evolve Guidance

Stein-guided SDXL sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
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

Batch outputs:

- One run directory per prompt under `--output-dir`.
- Per-run logs in `<output-dir>/_batch_logs` (`*.stdout.log`, `*.stderr.log`).

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

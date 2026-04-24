# Self-Improving Guidance

Stein-guided SDXL and SD sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
```

## Quick Start

Run one SDXL prompt:

```bash
python runs/single/gradient_sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sdxl
```

By default, post-generation evaluation is deferred. The run saves images and `final_rewards.json` metadata for later evaluation.
Use `--run-eval-now` if you want immediate final/trace scoring in the same run.

Run SDXL batch prompts:

```bash
python runs/gradient_sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --eval-reward image_reward \
  --devices cuda:0 cuda:1 \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --save-intermediate-rewards --trace-eval-batch 1 \
  --output-dir logs/sdxl_batch
```

For prompt-level parallelism, pass multiple GPUs with `--devices`. The batch runner will split the prompt list across those devices and run one worker process per GPU.

Run one SD 1.5 prompt:

```bash
python runs/single/gradient_sd.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sd
```

Run SD 1.5 batch prompts:

```bash
python runs/gradient_sd_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --eval-reward image_reward \
  --devices cuda:0 cuda:1 \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --save-intermediate-rewards --trace-eval-batch 1 \
  --output-dir logs/sd_batch
```

Useful flags:

- `--start-index 100 --max-prompts 50`: run a slice of prompts.
- `--stop-on-error`: stop on first failed prompt.
- `--dry-run`: print commands without running them.
- `--save-intermediate-images --trace-decode-batch-size 1`: save step images for each prompt.
- `--save-intermediate-rewards --trace-eval-batch 1`: save deferred intermediate reward traces.

Batch outputs:

- One run directory per prompt under `--output-dir`.
- Per-run logs in `<output-dir>/_batch_logs` (`*.stdout.log`, `*.stderr.log`).

SD default checkpoint:

- `runwayml/stable-diffusion-v1-5` (from `config/sd.py`)

### Main Options

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--prompt`: text prompt
- `--num-steps`: denoising steps
- `--num-particles`: particle count for Stein guidance
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein step size
- `--steer-start`, `--steer-end`: steering window (0-based step index)

### Outputs

SDXL saved in `logs/sdxl/<config>_seed<seed>`.

SD saved in `logs/sd/<config>_seed<seed>`.

Each run directory contains:

- Final particle images (`sample_*.png`)
- Final reward summary (`final_rewards.json`)
- Deferred reward traces (`steer_trace.csv`) when `--save-intermediate-rewards` is enabled
- Steer reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`) when enabled
- Optional eval reward plots (`eval_before_after_mean.png`, `eval_before_after_max.png`) when enabled

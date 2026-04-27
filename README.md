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
  --monitor-stein-delta \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sdxl
```

Run SDXL batch prompts:

```bash
python runs/gradient_sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --eval-reward image_reward \
  --device cuda \
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
  --monitor-stein-delta \
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
  --device cuda \
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
- `--monitor-stein-delta`: print per-step latent steering stats (`rel_delta`, `abs_delta`, `cosine_sim`).
- `--save-intermediate-images --trace-decode-batch-size 1`: save step images for each prompt.
- `--save-intermediate-rewards --trace-eval-batch 1`: save deferred intermediate reward traces.

Batch outputs:

- One run directory per prompt under `--output-dir`.
- Per-run logs in `<output-dir>/_batch_logs` (`*.stdout.log`, `*.stderr.log`).
- Batch summary CSV in `<output-dir>/batch_eval_summary.csv`.

SD default checkpoint:

- `runwayml/stable-diffusion-v1-5` (from `config/sd.py`)

### Main Options

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--prompt`: text prompt
- `--negative-prompt`: negative prompt text
- `--output-dir`: output root for artifacts
- `--device`: execution device (`cuda`, `cpu`, etc.)
- `--seed`: random seed override
- `--num-steps`: denoising steps
- `--batch-size`: number of base samples per prompt
- `--guidance-scale`: CFG strength
- `--eta`: DDIM eta
- `--num-particles`: particle count for Stein guidance
- `--batch-p`: reward-gradient micro-batch size over particles
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein step size
- `--stein-kernel`: Stein kernel (`rbf`)
- `--stein-adagrad-eps`: AdaGrad epsilon for Stein step adaptation
- `--kl-coeff`: reward scaling denominator
- `--monitor-stein-delta`: print per-step latent delta diagnostics (`rel_delta`, `abs_delta`, `cosine_sim`)
- `--steer-start`, `--steer-end`: steering window (0-based step index)

### Batch-only Options

- `--prompts-file`: input prompts file (`.txt` or `.json`)
- `--gradient-script` / `--sd-script` / `--sdxl-script`: single-run script path override
- `--python`: python executable used for each spawned run
- `--start-index`, `--max-prompts`: run a prompt slice
- `--stop-on-error`: stop on first failing run
- `--dry-run`: print generated commands only
- `--log-dir`: override batch log directory
- `--trace-decode-batch-size`: decode micro-batch size for intermediate image saving
- `--trace-eval-batch`: decode/eval micro-batch size for deferred reward traces
- `--intermediate-max-samples`: max samples per step for intermediate image dumps

### Outputs

SDXL saved in `logs/sdxl/<config>_seed<seed>`.

SD saved in `logs/sd/<config>_seed<seed>`.

Each run directory contains:

- Final particle images (`sample_*.png`)
- Final reward summary (`final_rewards.json`)
- Deferred reward traces (`steer_trace.csv`) when `--save-intermediate-rewards` is enabled
- Steer reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`) when enabled
- Optional eval reward plots (`eval_before_after_mean.png`, `eval_before_after_max.png`) when enabled

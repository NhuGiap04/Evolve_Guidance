# Self-Improving Guidance

Stein-guided SDXL and SD sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
```

## Quick Start

### SDXL

#### Gradient Stein Guidance

Single prompt:

```bash
python runs/single/gradient_sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --monitor-status \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sdxl
```

Batch prompts:

```bash
python runs/gradient/sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --device cuda \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --verbose --trace-eval-batch 1 \
  --output-dir logs/sdxl_batch
```

#### Approximate Guidance

Single prompt:

```bash
python runs/single/approx_sdxl.py \
  --config image_reward \
  --prompt "A cinematic portrait of a fox astronaut" \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --prediction-model default \
  --predicted-samples 1 \
  --monitor-status \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sdxl_approx
```

Batch prompts:

```bash
python runs/approx/sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config image_reward \
  --device cuda \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --prediction-model default \
  --predicted-samples 1 \
  --steer-start 0 \
  --steer-end 20 \
  --verbose --trace-eval-batch 1 \
  --output-dir logs/sdxl_approx_batch
```

### Stable Diffusion 1.5

#### Gradient Stein Guidance

Single prompt:

```bash
python runs/single/gradient_sd.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --monitor-status \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sd
```

Batch prompts:

```bash
python runs/gradient/sd_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --device cuda \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 0 \
  --steer-end 20 \
  --verbose --trace-eval-batch 1 \
  --output-dir logs/sd_batch
```

#### Approximate Guidance

Single prompt:

```bash
python runs/single/approx_sd.py \
  --config image_reward \
  --prompt "A cinematic portrait of a fox astronaut" \
  --num-steps 100 \
  --num-particles 4 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --prediction-model default \
  --predicted-samples 1 \
  --monitor-status \
  --steer-start 0 \
  --steer-end 20 \
  --output-dir logs/sd_approx
```

Batch prompts:

```bash
python runs/approx/sd_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config image_reward \
  --device cuda \
  --num-steps 100 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --prediction-model default \
  --predicted-samples 1 \
  --steer-start 0 \
  --steer-end 20 \
  --verbose --trace-eval-batch 1 \
  --output-dir logs/sd_approx_batch
```

Useful flags:

- `--start-index 100 --max-prompts 50`: run a slice of prompts.
- `--stop-on-error`: stop on first failed prompt.
- `--dry-run`: print commands without running them.
- `--monitor-status`: print per-step latent steering stats. Gradient runners report reward-gradient diagnostics; approx runners report soft good-score diagnostics.
- `--verbose --trace-eval-batch 1`: save deferred reward traces and control trace decode/eval batching.

Batch outputs:

- One run directory per prompt under `--output-dir`.
- Per-run logs in `<output-dir>/_batch_logs` (`*.stdout.log`, `*.stderr.log`).
- Batch summary CSV in `<output-dir>/batch_eval_summary.csv` with steering stats plus final stats for `clip`, `pick`, `image_reward`, `aesthetic`, and `hpsv2`.

SD default checkpoint:

- `runwayml/stable-diffusion-v1-5` (from `config/sd.py`)

### Main Options

- `--config`: reward preset (`pick`, `clip`, `image_reward`, `aesthetic`, `hpsv2`)
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
- `--batch-p`: reward evaluation / reward-gradient micro-batch size over particles
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein step size
- `--stein-kernel`: Stein kernel (`rbf`)
- `--stein-adagrad-eps`: AdaGrad epsilon for Stein step adaptation
- `--kl-coeff`: reward scaling denominator
- `--prediction-model`: approx-only clean prediction backend (`default`, `dpm`, `lcm`, `dmd`; only `default` is implemented currently)
- `--predicted-samples`: approx-only number of predicted clean samples per particle
- `--monitor-status`: print per-step latent delta diagnostics
- `--steer-start`, `--steer-end`: steering window (0-based step index)

### Batch-only Options

- `--prompts-file`: input prompts file (`.txt` or `.json`)
- `--gradient-script`, `--approx-script`, `--sd-script`, `--sdxl-script`: single-run script path override
- `--python`: python executable used for each spawned run
- `--start-index`, `--max-prompts`: run a prompt slice
- `--stop-on-error`: stop on first failing run
- `--dry-run`: print generated commands only
- `--log-dir`: override batch log directory
- `--trace-eval-batch`: decode/eval micro-batch size for deferred reward traces

### Outputs

SDXL gradient examples save in `logs/sdxl/<config>_seed<seed>`.

SD gradient examples save in `logs/sd/<config>_seed<seed>`.

Each run directory contains:

- Final particle images (`sample_*.png`)
- Final reward summary (`final_rewards.json`) including steering rewards and final-particle scores for `clip`, `pick`, `image_reward`, `aesthetic`, and `hpsv2`
- Deferred reward traces (`steer_trace.csv`) when `--verbose` is enabled
- Steer reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`) when enabled

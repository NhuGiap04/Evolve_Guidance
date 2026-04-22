# Self-Improving Guidance

Stein-guided SDXL sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
```

## Quick Start

Run one prompt:

```bash
python examples/gradient_sdxl.py \
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

Run Multiple Prompts

```bash
python runs/gradient_sdxl_batch.py \
  --prompts-file prompts/hps_v2_all_eval.txt \
  --config pick \
  --eval-reward image_reward \
  --device cuda \
  --num-steps 80 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.005 \
  --steer-start 20 \
  --steer-end 60 \
  --save-intermediate-rewards --trace-eval-batch 1 \
  --output-dir logs/sdxl_batch
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

### Main Options

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--prompt`: text prompt
- `--num-steps`: denoising steps
- `--num-particles`: particle count for Stein guidance
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein step size
- `--steer-start`, `--steer-end`: steering window (0-based step index)

### Outputs

Saved in `logs/sdxl/<config>_seed<seed>`:

- Final particle images (`sample_*.png`)
- Final reward summary (`final_rewards.json`)
- Deferred reward traces (`steer_trace.csv`) when `--save-intermediate-rewards` is enabled
- Steer reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`) when enabled
- Optional eval reward plots (`eval_before_after_mean.png`, `eval_before_after_max.png`) when enabled

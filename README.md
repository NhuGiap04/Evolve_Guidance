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
- Reward traces (`steer_pre_mean.npy`, `steer_post_mean.npy`, `steer_pre_max.npy`, `steer_post_max.npy`)
- Reward plots (`steer_before_after_mean.png`, `steer_before_after_max.png`)


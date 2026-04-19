# Self-Evolve Guidance

Improve diversity and quality of reward-based guidance methods by interacting and improving the particles population.

## Installation

```bash
pip install -e .
pip install --no-deps image-reward
```

## Example Usage

Run the detailed SDXL Stein-guided sampler:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --num-steps 80 \
  --guidance-scale 6.0 \
  --num-particles 4 \
  --stein-loop 2 \
  --stein-step 0.02 \
  --steer-start 0 \
  --steer-end 79 \
  --output-dir logs/sdxl
```

Run with intermediate decoded images at each denoising step:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --eval-reward image_reward \
  --save-intermediate-images \
  --trace-decode-batch-size 2 \
  --intermediate-max-samples 2 \
  --output-dir logs/sdxl
```

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--num-particles`: number of particles used in Stein updates
- `--stein-loop`: number of inner Stein loops per steered timestep
- `--stein-step`: base step size for Stein AdaGrad updates
- `--eval-reward`: optional final-image eval scorer (`none`, `clip`, `pick`, `image_reward`)
- `--steer-start`, `--steer-end`: steering window over inference-step indices (0-based, defaults to full range)
- `--trace-decode-batch-size`: decode micro-batch size for intermediate image tracing
- `--save-intermediate-images`: save decoded intermediate images for each denoising step
- `--intermediate-max-samples`: optional cap on number of saved intermediate samples per step

The script now prints before/after steering rewards to terminal for each steered step:

- mean reward: `pre -> post` with delta
- max reward: `pre -> post` with delta

Outputs are saved under `logs/sdxl_detailed/<config>_seed<seed>`.

Key visualization outputs:

- `steer_before_after_mean.png`: before vs after steering mean reward curve
- `steer_before_after_max.png`: before vs after steering max reward curve
- `steer_pre_mean.npy`, `steer_post_mean.npy`
- `steer_pre_max.npy`, `steer_post_max.npy`
- final sampled images named with steering score

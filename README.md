# Self-Evolve Guidance

Improve diversity and quality of reward-based guidance methods by interacting and improving the particles population.

## Installation

```bash
pip install -e .
pip install --no-deps image-reward
```

## Usage

### 1) Basic run (recommended starting point)

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --negative-prompt "blurry, low quality, distorted" \
  --eval-reward image_reward \
  --device cuda \
  --seed 42 \
  --num-steps 80 \
  --guidance-scale 6.0 \
  --eta 1.0 \
  --batch-size 1 \
  --num-particles 4 \
  --batch-p 1 \
  --stein-loop 1 \
  --stein-step 0.02 \
  --stein-adagrad-eps 1e-8 \
  --kl-coeff 1.0 \
  --steer-start 20 \
  --steer-end 60 \
  --output-dir logs/sdxl
```

### 2) Save intermediate decoded images for each denoising step

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

Then run the same command as above. For each steered timestep, you will see:

- `pre_mean -> post_mean`
- `pre_max -> post_max`

## Important Arguments

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--eval-reward`: optional final-image eval scorer (`none`, `clip`, `pick`, `image_reward`)
- `--num-particles`: number of particles for Stein guidance
- `--batch-p`: reward-gradient micro-batch in particle units
- `--stein-loop`: number of Stein inner updates per steered step
- `--stein-step`: base step size for Stein updates
- `--kl-coeff`: scales reward term in the score combination
- `--steer-start`, `--steer-end`: steering window over inference-step indices (0-based)
- `--save-intermediate-images`: save decoded intermediate images for each denoising step
- `--trace-decode-batch-size`: decoding micro-batch size while tracing
- `--intermediate-max-samples`: optional cap on saved intermediate samples per step


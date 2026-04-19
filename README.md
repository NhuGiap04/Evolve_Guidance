# Self-Evolve Guidance

Improve diversity and quality of reward-based guidance methods by interacting and improving the particles population.

## Installation

```bash
pip install -e .
pip install --no-deps image-reward
```

## Example Usage

Run the SDXL Stein-guided sampler:

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --num-steps 80 \
  --guidance-scale 6.0 \
  --num-particles 4 \
  --stein-loop 2 \
  --stein-step 0.02 \
  --steer-start 0 \
  --steer-end 79 \
  --save-logs \
  --output-dir logs/examples/sdxl_stein
```

Fast run (skip intermediate logs/plots, but still save final images and rewards):

```bash
python examples/sdxl.py \
  --config pick \
  --prompt "A cinematic portrait of a fox astronaut" \
  --no-save-logs \
  --output-dir logs/examples/sdxl_stein
```

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--num-particles`: number of particles used in Stein updates
- `--stein-loop`: number of inner Stein loops per steered timestep
- `--stein-step`: base step size for Stein AdaGrad updates
- `--steer-start`, `--steer-end`: steering window over inference-step indices (0-based, defaults to full range)
- `--save-logs` / `--no-save-logs`: toggle intermediate reward logs and plots
- `--save-intermediate-images`: optionally save before/after steered intermediate images

Outputs are saved under `logs/examples/sdxl_stein/<config>_seed<seed>`.

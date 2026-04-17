# Self-Evolve Guidance

Improve diversity and quality of reward-based guidance methods by interacting and improving the particles population.

## Installation

```bash
pip install -e .
pip install --no-deps image-reward
```

## Example Usage

Run the detailed SDXL example with intermediate reward tracing and plotting:

```bash
python examples/sdxl.py --config pick --prompt "A cinematic portrait of a fox astronaut" --num-steps 80 --guidance-scale 6.0 --eval-reward image_reward --output-dir logs/examples/sdxl_detailed --trace-decode-batch-size 4
```

- `--config`: reward preset (`pick`, `clip`, `seg`)
- `--prompt`: text prompt for generation
- `--num-steps`: number of denoising steps
- `--guidance-scale`: CFG strength
- `--eval-reward`: optional evaluation trace scorer (`none`, `clip`, `pick`, `image_reward`)
- `--output-dir`: folder for images and trace files

Outputs are saved under `logs/examples/sdxl_detailed/<config>_seed<seed>`.

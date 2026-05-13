# Self-Improving Guidance

Stein-guided SDXL and SD sampling for reward optimization.

## Setup

```bash
pip install -e .
pip install --no-deps image-reward
```

For TCE with Francisco Ibarrola's `image_diversity` implementation:

```bash
pip install -e ".[tce]"
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

## GenEval metrics

This repo can run the GenEval object-based evaluation on existing batch outputs.

Setup (one time):

```bash
pip install -e ".[geneval]"
git clone https://github.com/djghosh13/geneval.git
cd geneval
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
```

Run evaluation on a batch root:

```bash
python eval_geneval_outputs.py \
  --eval-root logs/sd_batch/batch_20260424_111205_6101 \
  --geneval-prompts path/to/geneval/prompts/evaluation_metadata.jsonl \
  --geneval-repo path/to/geneval \
  --geneval-model-path <OBJECT_DETECTOR_FOLDER> \
  --samples-per-prompt 4
```

Notes:
- GenEval requires CUDA and Mask2Former weights downloaded via `download_models.sh`.
- The script matches prompts from `final_rewards.json` against the GenEval prompt set.

## TCE metrics

This repo can calculate Truncated CLIP Entropy (TCE) on generated `sample_*.png`
sets with `fibarrola/image_diversity`:

```bash
python calculate_tce.py \
  --eval-root logs/sd_batch/batch_20260424_111205_6101 \
  --n-eigs 20 \
  --batch-size 16
```

To calculate one TCE over the best image from each prompt/run:

```bash
python calculate_tce.py \
  --eval-root logs/sd_batch/batch_20260424_111205_6101 \
  --best-per-prompt \
  --n-eigs 20 \
  --batch-size 16
```

By default, `--best-per-prompt` chooses the best image using `eval_rewards`
from `final_rewards.json`, then falls back to `steer_rewards`. To choose from
an offline evaluator score column instead, run the evaluator first and pass,
for example, `--best-score-column pick`.

Outputs:
- Per-run `tce_image_diversity.csv` inside each run directory.
- Batch-level `tce_image_diversity_batch.csv`.
- Batch-level `tce_image_diversity_summary.csv`.
- With `--best-per-prompt`: `tce_best_per_prompt_selection.csv`,
  `tce_best_per_prompt.csv`, and staged selected images in
  `tce_best_per_prompt_images/`.

The script clamps `--n-eigs` to `num_images - 1` for each run because
`image_diversity` requires the truncation count to be smaller than the number
of images. You can also use the same backend from the full evaluator:

```bash
python eval_generated_outputs.py \
  --eval-root logs/sd_batch/batch_20260424_111205_6101 \
  --run-diversity \
  --diversity-backend image_diversity
```

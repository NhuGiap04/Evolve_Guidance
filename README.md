# Self-Evolving Guidance

Experiment code for self-evolving test-time guidance methods. The repository
contains Stein-guided text-to-image diffusion experiments, offline RL flow
guidance baselines, and a discrete diffusion workspace for text generation
guidance and evaluation. Each experiment family keeps its runners, models, and
setup notes in a separate section.

## Repository Map

- `text2img/`: text-to-image configs, runners, reward models, scorers, prompts, and patched diffusion pipelines.
- `offline_rl/`: Guided Flow Planner experiments for D4RL locomotion tasks.
- `discrete_diffusion/`: discrete diffusion migration workspace and evaluation tools.

## Experiments

### Text-To-Image Stein Guidance

#### Setup
```bash
pip install -e ./text2img
```

Optional reward dependencies:

- HPSv2: install from `https://github.com/tgxs002/HPSv2`
- ImageReward: `pip install --no-deps image-reward`

Entry points:

| Script | Model | Guidance |
| --- | --- | --- |
| `text2img/runs/grad_sd.py` | Stable Diffusion 1.5 | reward-gradient Stein |
| `text2img/runs/grad_sdxl.py` | SDXL | reward-gradient Stein |
| `text2img/runs/approx_sd.py` | Stable Diffusion 1.5 | approximate reward Stein |
| `text2img/runs/approx_sdxl.py` | SDXL | approximate reward Stein |

#### Running Experiments

After installation, you can run the scripts from inside `text2img/`:

```bash
cd text2img
python runs/grad_sd.py
python runs/grad_sdxl.py
python runs/approx_sd.py
python runs/approx_sdxl.py
```

Use `--prompts-file` for batch experiments.

#### Important Arguments

- `--prompts-file`: input prompts file (`.txt` or `.json`)
- `--config`: reward preset, one of `pick`, `clip`, `image_reward`, `aesthetic`, `hpsv2`
- `--device`: execution device, for example `cuda`
- `--output-dir`: output root
- `--seed`: random seed override
- `--num-steps`: denoising steps
- `--num-particles`: particle count
- `--batch-p`: reward micro-batch size over particles
- `--stein-loop`: Stein updates per steered step
- `--stein-step`: Stein update size
- `--stein-kernel`: `rbf` or `imq`
- `--start`, `--end`: 0-based steering window, with `end` exclusive; defaults are `0` and `--num-steps`
- `--start-index`, `--max-prompts`: select a prompt slice in batch mode
- `--stop-on-error`: stop batch execution on the first failed prompt
- `--dry-run`: print planned runs without executing them
- `--prediction-model`: approx-only clean prediction backend
- `--predicted-samples`: approx-only number of predicted clean samples per particle
- `--lookahead-steps`: approx-only clean-prediction solver steps

#### Outputs

Batch runs write to:

```text
<output-dir>/run_<idx>_<prompt-slug>/<config>_seed<seed>/
<output-dir>/_batch_logs/
<output-dir>/batch_eval_summary.csv
```

Run directories contain generated images and `final_rewards.json`.

### Locomotion Flow Guidance

The `locomotion/` folder contains the Guided Flow Planner (`gflower`) offline RL
experiments adapted from the flow guidance baseline. These runs train flow
matching trajectory models on D4RL locomotion tasks, train value models, and
evaluate value-gradient guidance plus trajectory-particle Stein steering.

#### Setup

The locomotion stack uses older MuJoCo/D4RL dependencies. Change into the
`locomotion/` folder and create the pinned environment:

```bash
cd locomotion
conda env create -f environment.yml
conda activate gflower
pip install -e .
```

MuJoCo 2.1 is required by `mujoco-py`:

```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```

If `mujoco-py` fails to compile, install GL/GCC support from conda-forge and
rebuild with the conda compiler:

```bash
conda install -c conda-forge gcc glew mesalib -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LDSHARED="$CC -shared"
export CFLAGS="-Wno-error=incompatible-pointer-types"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"
pip install --force-reinstall --no-build-isolation --no-cache-dir "cython<3" "mujoco-py==2.1.2.14"
```

D4RL locomotion datasets are downloaded automatically to `~/.d4rl` when the
training scripts first access them.

#### Running Experiments

Run these commands from inside `locomotion/` after activating `gflower`:

```bash
bash run_scripts/train.sh
bash run_scripts/train_value.sh
bash run_scripts/eval_gradient.sh
bash run_scripts/eval_stein.sh
```

Script roles:

- `train.sh`: train CFM and OT-CFM base trajectory flow models.
- `train_value.sh`: train the value model used by guided evaluation.
- `eval_gradient.sh`: evaluate value-gradient guidance variants.
- `eval_stein.sh`: evaluate grouped RBF Stein steering over trajectory
  particles, using the value-gradient direction as the target score.

These scripts sweep over `halfcheetah`, `hopper`, and `walker2d` D4RL datasets.
For quick smoke tests, reduce the loops in the shell scripts or run the
corresponding `python run/*.py` command directly with fewer training steps.

### Discrete Diffusion

This area is prepared for future discrete diffusion experiments.

Current utilities:

- `discrete_diffusion/evaluation/mdlm_to_eval_format.py`: convert MDLM-style samples to SSD-LM evaluation format.
- `discrete_diffusion/evaluation/compute_metrics.sh`: compute PPL, CoLA, diversity, and toxicity metrics for generated JSONL files.
- `discrete_diffusion/evaluation/aggregate_over_seeds_mdlm.py`: aggregate metric files across seeds or repeated runs.
- `discrete_diffusion/reward_functions.py`: reward/model scoring utilities.

# Stein-Guided Test-Time Alignment for Diffusion and Flow Models

Experiment code for Stein-guided sampling and test-time alignment methods. The repository
contains Stein-guided text-to-image diffusion experiments, offline RL flow
guidance baselines and Stein test-time alignment for image inverse problems.
Each experiment family keeps its runners, models, and setup notes in a separate
section.

## Repository Map

- `text2img/`: text-to-image configs, runners, reward models, scorers, prompts, and patched diffusion pipelines.
- `inverse/`: Stein test-time alignment experiments for image inverse problems with flow-matching priors.
- `locomotion/`: Guided Flow Planner experiments for D4RL locomotion tasks.

## Experiments

### 1. Text-To-Image Stein Guidance

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

#### Running Experiments

After installation, you can run the scripts from inside `text2img/`:

```bash
cd text2img
python runs/grad_sd.py
python runs/grad_sdxl.py
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
- `--stein-step`: integrated Stein coefficient (`lambda_t`) per diffusion step
- `--stein-kernel`: `rbf`,`imq`, `vmf`
- `--stein-repulsion`: Stein repulsion strength
- `--repulsion-schedule`: `const` or `linear_decay`; the linear schedule increases
  repulsion from zero at the initial noise to the configured strength at the end
- `--start`, `--end`: 0-based steering window, with `end` exclusive; defaults are `0` and `--num-steps`
- `--start-index`, `--max-prompts`: select a prompt slice in batch mode
- `--stop-on-error`: stop batch execution on the first failed prompt
- `--dry-run`: print planned runs without executing them

At each selected denoising step, the SD/SDXL scheduler supplies the base drift
`u_t`. The pipeline adds the grouped particle field
`mean_j[k(x_j,x_i) grad log(h_t(x_j)) + gamma_t grad_(x_j) k(x_i,x_j)]`,
where `log(h_t) = reward / kl_coeff` and `gamma_t` is controlled by
`stein_repulsion` and `repulsion_schedule`. This field is evaluated exactly
once per selected diffusion step. The SD and SDXL presets use `eta=0` for
deterministic ODE dynamics; passing a nonzero `--eta` remains supported as a
stochastic DDIM extension.

#### Outputs

Batch runs write to:

```text
<output-dir>/run_<idx>_<prompt-slug>/<config>_seed<seed>/
<output-dir>/_batch_logs/
<output-dir>/results.csv
```

Run directories contain generated images and `final_rewards.json`.

### 2. Image Inverse Problems

The `inverse/` module tests test-time alignment of flow-matching image models
for noisy inverse problems. It supports inpainting, Gaussian deblurring, and
super-resolution, with PiGDM/PiGDM+, gradient-based, and Monte Carlo guidance
variants.

#### Setup

Install the module and its dependencies from the repository root:

```bash
cd inverse
pip install -r requirements.txt
pip install -e .
```

The current data loader expects CelebA-HQ images under
`inverse/data_cache/celeba_hq_256/`. Place the matching flow-model checkpoint
under `inverse/results/`; for example, the default configuration loads:

```text
inverse/results/cfm_punet256_celeba256/model_499.pt
```

#### Running Experiments

Run inverse-problem experiments from inside `inverse/`. This example uses the
default 256x256 CelebA-HQ model for inpainting on GPU 0:

```bash
cd inverse
python run/inference_inverse.py \
  --device cuda:0 \
  --data_cache_dir . \
  --problem inpainting \
  --guide_method PiGDM
```

Set `--problem` to `inpainting`, `deblurring`, or `superresolution`. Supported
guidance methods are `PiGDM`, `PiGDM+`, `nabla_xt_J_xt`, `nabla_x1_J_x1`,
`nabla_xt_J_x1`, and `MC`. The scripts in `inverse/scripts/` provide parameter
sweeps for the available guidance variants:

```bash
bash scripts/PiGDM.sh
bash scripts/g_sim_inv_A.sh
bash scripts/g_cov_A.sh
bash scripts/g_cov_G.sh
bash scripts/g_MC.sh
```

Each run writes generated reconstructions, ground-truth images, degraded
measurements, and `metrics.txt` under `inverse/infer/<experiment-name>/`.

### 3. Locomotion Flow Guidance

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
bash run_scripts/eval_smc.sh
bash run_scripts/eval_gradient.sh
bash run_scripts/eval_stein.sh
```

Script roles:

- `train.sh`: train CFM and OT-CFM base trajectory flow models.
- `train_value.sh`: train the value model used by guided evaluation.
- `eval_smc.sh`: evaluate value-weighted sequential Monte Carlo with adaptive
  effective-sample-size resampling.
- `eval_gradient.sh`: evaluate value-gradient guidance variants.
- `eval_stein.sh`: evaluate grouped RBF Stein steering over trajectory
  particles, using the value-gradient direction as the target score.

These scripts sweep over `halfcheetah`, `hopper`, and `walker2d` D4RL datasets.
For quick smoke tests, reduce the loops in the shell scripts or run the
corresponding `python run/*.py` command directly with fewer training steps.

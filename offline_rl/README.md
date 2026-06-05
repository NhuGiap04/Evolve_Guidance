# Offline RL Flow Guidance

This folder contains the Guided Flow Planner (`gflower`) offline RL experiments
for D4RL locomotion tasks. The workflow is:

1. Train a trajectory flow model.
2. Train a value model.
3. Evaluate unguided, value-gradient guided, or Stein particle guided planning.

The default scripts target the `halfcheetah`, `hopper`, and `walker2d` D4RL
`medium`, `medium-replay`, and `medium-expert` datasets.

For a focused local and Google Colab setup walkthrough, see
[`SETUP_README.md`](SETUP_README.md).

## Requirements

- Linux environment with CUDA for the default scripts.
- Conda.
- MuJoCo 2.1 for `mujoco-py`.
- Enough disk for D4RL datasets and logs. Datasets are downloaded on first use
  to `~/.d4rl`.

The environment is pinned in `environment.yml` and uses Python 3.8 plus older
MuJoCo/D4RL dependencies.

## Local Setup

Run these commands from the repository root:

```bash
cd offline_rl
conda env create -f environment.yml
conda activate gflower
pip install -e .
```

Install MuJoCo 2.1:

```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```

If `mujoco-py` fails to build, install the compiler and OpenGL dependencies
inside the conda environment, then rebuild `mujoco-py`:

```bash
conda install -c conda-forge gcc glew mesalib -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LDSHARED="$CC -shared"
export CFLAGS="-Wno-error=incompatible-pointer-types"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"
pip install --force-reinstall --no-build-isolation --no-cache-dir "cython<3" "mujoco-py==2.1.2.14"
```

Check the install:

```bash
python - <<'PY'
import gym
import d4rl
import mujoco_py

env = gym.make("hopper-medium-v2")
print("Loaded:", env.spec.id)
PY
```

If `conda activate gflower` is unavailable in a shell script, initialize conda
for the current shell first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gflower
```

## Google Colab Setup

Use a GPU runtime. In Colab, install conda with `condacolab` first. This cell
restarts the runtime when it finishes:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

After the runtime restarts, clone the repository or mount your Drive copy, then
create the environment:

```python
import condacolab
condacolab.check()
```

```bash
git clone <your-repo-url>
cd Evolve_Guidance/offline_rl
conda env create -f environment.yml
conda run -n gflower pip install -e .
```

Install MuJoCo 2.1 and system libraries in Colab:

```bash
apt-get update
apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Run project commands inside the `gflower` environment. In Colab notebook shell
cells, either use `conda run -n gflower ...` or use one `%%bash` cell that
activates the environment before running commands:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
conda run -n gflower python - <<'PY'
import gym
import d4rl
import mujoco_py

env = gym.make("hopper-medium-v2")
print("Loaded:", env.spec.id)
PY
```

For long Colab runs, write logs to Google Drive so checkpoints survive runtime
disconnects:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then pass a Drive-backed log folder, for example:

```bash
--log_folder /content/drive/MyDrive/evolve_guidance_offline_rl_logs
```

## Full Experiment Scripts

Run from `offline_rl/` after activating `gflower` locally, or inside a Colab
shell that has activated the environment:

```bash
bash run_scripts/train.sh
bash run_scripts/train_value.sh
bash run_scripts/eval_gradient.sh
bash run_scripts/eval_stein.sh
```

Script roles:

- `run_scripts/train.sh`: trains CFM and OT-CFM trajectory flow models.
- `run_scripts/train_value.sh`: trains the value model for guided evaluation.
- `run_scripts/eval_gradient.sh`: evaluates value-gradient guidance variants.
- `run_scripts/eval_stein.sh`: evaluates grouped RBF Stein steering over
  trajectory particles.

The full flow training script uses `--n_train_steps 1000001` and can take a long
time. Evaluation scripts expect checkpoints produced by the training scripts:

- Flow checkpoint: `logs/<env>/flow/<flow_exp_name>/model_ema_<flow_cp>.pth`
- Value checkpoint: `logs/<env>/value/<value_exp_name>/model_<value_cp>.pth`

## Quick Smoke Test

These commands run a tiny Hopper job and are intended only to verify that the
environment, dataset loading, checkpoint writing, and evaluation path work.

```bash
python run/train.py \
  --device cuda:0 \
  --log_folder ./logs_smoke \
  --exp_name H20_smoke \
  --env hopper-medium-v2 \
  --horizon 20 \
  --state_dim 11 \
  --action_dim 3 \
  --n_train_steps 2 \
  --save_freq 1 \
  --lr_schdule_T 2 \
  --batch_size 2 \
  --flow_matching_type cfm
```

```bash
python run/train_value.py \
  --device cuda:0 \
  --log_folder ./logs_smoke \
  --exp_name H20_smoke_value \
  --env hopper-medium-v2 \
  --inf_horizon \
  --horizon 20 \
  --state_dim 11 \
  --action_dim 3 \
  --n_train_steps 2 \
  --save_freq 1 \
  --batch_size 2
```

```bash
python run/eval.py \
  --device cuda:0 \
  --log_folder ./logs_smoke \
  --seed 0 \
  --random_repeat 1 \
  --max_episode_length 5 \
  --exp_name H20_smoke_gradient_eval \
  --env hopper-medium-v2 \
  --state_dim 11 \
  --action_dim 3 \
  --horizon 20 \
  --flow_exp_name H20_smoke \
  --flow_cp 1 \
  --flow_matching_type cfm \
  --value_exp_name H20_smoke_value \
  --value_cp 1 \
  --ode_t_steps 2 \
  --guidance_method gradient \
  --grad_compute_at x_1 \
  --grad_wrt x_1 \
  --grad_schedule const \
  --grad_scale 0.01
```

On Colab, prefix those commands with `conda run -n gflower` and keep the MuJoCo
environment variables exported in the same shell cell.

## Common Direct Commands

Train one flow model:

```bash
python run/train.py \
  --device cuda:0 \
  --log_folder ./logs \
  --exp_name H20_1e6steps \
  --env hopper-medium-v2 \
  --horizon 20 \
  --state_dim 11 \
  --action_dim 3 \
  --n_train_steps 1000001 \
  --save_freq 50000 \
  --lr_schdule_T 1000000 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --ema_decay 0.995 \
  --flow_matching_type cfm
```

Train one value model:

```bash
python run/train_value.py \
  --device cuda:0 \
  --log_folder ./logs \
  --exp_name H20_inf \
  --env hopper-medium-v2 \
  --inf_horizon \
  --horizon 20 \
  --state_dim 11 \
  --action_dim 3 \
  --n_train_steps 10001 \
  --save_freq 5000 \
  --batch_size 64 \
  --learning_rate 2e-4
```

Evaluate one Stein-guided run after the matching flow and value checkpoints
exist:

```bash
python run/eval.py \
  --device cuda:0 \
  --seed 0 \
  --random_repeat 5 \
  --exp_name H20_1e6steps_stein_10steps_inf_K8_loop1_step0.02_scale0.1 \
  --env hopper-medium-v2 \
  --state_dim 11 \
  --action_dim 3 \
  --horizon 20 \
  --flow_exp_name H20_1e6steps \
  --flow_cp 19 \
  --flow_matching_type cfm \
  --value_exp_name H20_inf \
  --value_cp 2 \
  --ode_t_steps 10 \
  --guidance_method stein \
  --grad_compute_at x_1 \
  --grad_wrt x_1 \
  --grad_schedule const \
  --grad_scale 0.1 \
  --stein_particles 8 \
  --stein_loop 1 \
  --stein_step 0.02 \
  --stein_kernel rbf
```

## Output Layout

Outputs are written under `--log_folder`, which defaults to `logs`:

```text
logs/<env>/flow/<exp_name>/
logs/<env>/value/<exp_name>/
logs/<env>/eval/<exp_name>/
```

Flow directories contain model checkpoints, EMA checkpoints, TensorBoard logs,
and saved config files. Value directories contain value checkpoints and
TensorBoard logs. Evaluation directories contain the saved config and
`results.txt`.

## Notes

- The scripts default to `cuda:0`. For CPU-only checks, pass `--device cpu`, but
  full experiments are intended for GPU.
- D4RL MuJoCo datasets are downloaded automatically on first use.
- The `state_dim` and `action_dim` values used by the scripts are:
  `hopper`: `11/3`, `halfcheetah`: `17/6`, `walker2d`: `17/6`.
- `ot_cfm` runs use the `ot_` prefix in the provided scripts, for example
  `ot_H20_1e6steps`.

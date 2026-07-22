# Offline RL Setup Guide

This guide covers two ways to run the `offline_rl` experiments:

- Local Linux machine with Conda and CUDA.
- Google Colab GPU runtime with `condacolab`.

Run commands from the repository root unless a step explicitly says to `cd`
elsewhere.

## What This Project Needs

The offline RL code uses older D4RL and `mujoco-py` dependencies, so the setup is
pinned to Python 3.8 and MuJoCo 2.1.

Main requirements:

- Linux environment.
- CUDA GPU for full experiments.
- Conda or `condacolab`.
- MuJoCo 2.1 installed at `~/.mujoco/mujoco210`.
- Disk space for D4RL datasets, downloaded on first use to `~/.d4rl`.
- Disk space for logs and checkpoints under `offline_rl/logs` unless
  `--log_folder` is changed.

## Local Setup

Create and activate the Conda environment:

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
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Export the MuJoCo variables in every shell that runs training or evaluation:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```

If `conda activate gflower` does not work in a script, initialize Conda first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gflower
```

Check the installation:

```bash
python - <<'PY'
import gym
import d4rl
import mujoco_py

env = gym.make("hopper-medium-v2")
print("Loaded:", env.spec.id)
PY
```

If `mujoco-py` fails to build, install compiler and OpenGL dependencies inside
the environment, then reinstall `mujoco-py`:

```bash
conda install -c conda-forge gcc glew mesalib -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LDSHARED="$CC -shared"
export CFLAGS="-Wno-error=incompatible-pointer-types"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"
pip install --force-reinstall --no-build-isolation --no-cache-dir "cython<3" "mujoco-py==2.1.2.14"
```

## Google Colab Setup

Use a GPU runtime:

```text
Runtime -> Change runtime type -> T4 GPU or better
```

Install Conda through `condacolab`. This cell restarts the runtime after it
finishes:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

After the runtime restarts, verify Conda:

```python
import condacolab
condacolab.check()
```

Clone the repository, or mount Drive if your copy is already there:

```bash
git clone <your-repo-url>
cd Evolve_Guidance/offline_rl
```

Create the project environment and install the package:

```bash
conda env create -f environment.yml
conda run -n gflower pip install -e .
```

Install MuJoCo 2.1 and the system libraries needed by `mujoco-py`:

```bash
apt-get update
apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

In Colab, prefer `conda run -n gflower ...` because each notebook shell cell is
a fresh shell. Keep the MuJoCo exports in the same cell as the command:

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

For long runs, mount Google Drive and write logs there so checkpoints survive
runtime disconnects:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Pass a Drive-backed log folder to training and evaluation commands:

```bash
--log_folder /content/drive/MyDrive/evolve_guidance_offline_rl_logs
```

## Smoke Test

Run a tiny Hopper flow training job:

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

Train a tiny value model:

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

Evaluate a short guided rollout:

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

On Colab, use the same commands with `conda run -n gflower` and the MuJoCo
environment variables in the same shell cell:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
conda run -n gflower python run/train.py --device cuda:0 ...
```

## Full Experiments

The provided scripts run all configured D4RL locomotion tasks:

```bash
bash run_scripts/train.sh
bash run_scripts/train_value.sh
bash run_scripts/eval_best_of_n.sh
bash run_scripts/eval_gradient.sh
bash run_scripts/eval_stein.sh
```

They expect to be launched from `offline_rl/`.

The main scripts are:

- `run_scripts/train.sh`: trains CFM and OT-CFM trajectory flow models.
- `run_scripts/train_value.sh`: trains value models.
- `run_scripts/eval_best_of_n.sh`: evaluates best-of-N trajectory selection
  for N in 1, 4, 8, 16, 32, and 64.
- `run_scripts/eval_gradient.sh`: evaluates value-gradient guidance.
- `run_scripts/eval_stein.sh`: evaluates grouped RBF Stein particle guidance.

Full training is long. The flow script uses one million training steps by
default, and evaluation expects matching checkpoints:

```text
logs/<env>/flow/<flow_exp_name>/model_ema_<flow_cp>.pth
logs/<env>/value/<value_exp_name>/model_<value_cp>.pth
```

## Common Issues

`No module named gflower`:

Run `pip install -e .` or `conda run -n gflower pip install -e .` from
`offline_rl/`.

`mujoco_py` cannot find MuJoCo:

Check that `~/.mujoco/mujoco210` exists and that `MUJOCO_PY_MUJOCO_PATH` points
to it.

OpenGL or `GL/glew.h` build errors:

Install the OpenGL and compiler dependencies listed above, then reinstall
`mujoco-py`.

CUDA not available:

Confirm that the runtime has a GPU and use `--device cuda:0`. For quick CPU-only
import checks, pass `--device cpu`, but full experiments are intended for CUDA.

D4RL dataset download is slow:

The first run downloads datasets to `~/.d4rl`. On Colab, this cache is lost when
the runtime is reset unless you copy it to Drive.

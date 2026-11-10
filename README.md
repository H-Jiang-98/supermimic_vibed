# SuperMimic

Unofficial SuperMimic reproduction for G1 motion tracking with Isaac Lab and MuJoCo-Warp.
See [implementation and attribution](docs/reference.md). Code: [MIT](LICENSE).

## Environment and paths

Use an **already configured Conda environment** with Isaac Lab, Newton/MuJoCo-Warp,
PyTorch and the asset importers installed. The local environment is `supermimic`.
Run commands from the repository root; `bash` and `python3` must be on `PATH`.

Edit [`config.json`](config.json):

```json
{
  "conda_env": "tools/activate.sh",
  "motion_dir": "data/lafan1_g1",
  "robot_description_dir": "data/g1_description",
  "robot_usd_dir": "data/g1_usd",
  "ground_dir": "data/ground",
  "logs_dir": "logs"
}
```

Paths may be absolute or relative to the config file. See [configuration details](docs/configuration.md)
for the expected asset layout. The `training` section in `config.json` includes all training
hyperparameters with comments showing code defaults. The file supports `//` and `/* */` comments
(JSONC). Command-line flags override configured values; omitted values use code defaults.
Launchers source `conda_env` before running the task. The supplied `tools/activate.sh` activates
`supermimic` through `~/miniconda3`; edit it or point `conda_env` to your own activation script.

```bash
# Skip downloading if the data and converted assets already exist.
python3 tools/run.py tools/download_data.sh
python3 tools/run.py tools/test_env.py --num_envs 40 --visualizer none
```

## Train

```bash
./run_train.sh lafan1_main --iterations 14000 --visualizer none

# Smaller starting configuration; adjust to available GPU memory.
./run_train.sh lafan1_small --num_envs 1024 --buffer_steps 500 \
  --wm_batch 512 --pi_batch 1024 --iterations 14000 --visualizer none
```

Training runs in the foreground. For a persistent terminal, run `tmux new -s lafan1_main`
first, then launch training; detach with Ctrl+B, D. Optional CPU pinning: `CPUS=0-23 ./run_train.sh ...`.
The first iterations compile kernels. The first 500 iterations train only the world model.
Logs, resolved configuration and checkpoints go to `<logs_dir>/<run>/` (`logs/<run>/` by default).

```bash
LOGS_DIR=$(python3 supermimic/config.py logs_dir)
tail -f "$LOGS_DIR/lafan1_main/stdout.log"

# Resume after the previous process exits; repeat your original configuration flags.
./run_train.sh lafan1_main --resume "$LOGS_DIR/lafan1_main/last.pt" \
  --iterations 28000 --visualizer none
```

`--iterations` is the target total iteration count. Resume from `last.pt`;
numbered `ckpt_*.pt` files are for evaluation.

### Independent experiments on multiple GPUs

On a machine with four GPUs, use `nvidia-smi` to check their indices, then run one observation
mode per GPU. All four experiments use the same seed and training settings:

```bash
# GPU 0: paper observation (93 dimensions).
CUDA_VISIBLE_DEVICES=0 ./run_train.sh obs_paper --obs_mode paper \
  --seed 0 --iterations 14000 --visualizer none &
# GPU 1: rotation error, SO(3) logarithm (96 dimensions).
CUDA_VISIBLE_DEVICES=1 ./run_train.sh obs_so3 --obs_mode so3 \
  --seed 0 --iterations 14000 --visualizer none &
# GPU 2: rotation and position error, SE(3) logarithm (99 dimensions).
CUDA_VISIBLE_DEVICES=2 ./run_train.sh obs_se3 --obs_mode se3 \
  --seed 0 --iterations 14000 --visualizer none &
# GPU 3: rotation, position and linear velocity error, SE₂(3) logarithm (102 dimensions).
CUDA_VISIBLE_DEVICES=3 ./run_train.sh obs_se23 --obs_mode se23 \
  --seed 0 --iterations 14000 --visualizer none &
wait
```

Each job trains its own model on one GPU; the selected GPU appears as `cuda:0` inside that process.
Logs and checkpoints go to `<logs_dir>/obs_paper/`, `obs_so3/`, `obs_se3/`, and `obs_se23/`.
If overriding `--log_dir`, also choose a different directory for each job.
The same `CUDA_VISIBLE_DEVICES` prefix works with `./run_eval.sh`.

## Evaluate and render

```bash
# Evaluate saved checkpoints. Add --watch to wait for future checkpoints.
./run_eval.sh lafan1_main --visualizer none
LOGS_DIR=$(python3 supermimic/config.py logs_dir)
python3 tools/run.py tools/plot.py "$LOGS_DIR/lafan1_main" --out results/success_curve.png

# Evaluate one checkpoint and record its trajectories.
./run_eval.sh lafan1_main \
  --ckpt "$LOGS_DIR/lafan1_main/ckpt_014000.pt" --record traj.npz --visualizer none
python3 tools/run.py tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

Evaluation writes `<logs_dir>/<run>/eval.csv` when scanning checkpoints. Rendering requires FFmpeg
and a working NVIDIA EGL driver installation.

To use another configuration, put `--config /path/to/config.json` immediately after
`./run_train.sh`, `./run_eval.sh`, or `python3 tools/run.py`.

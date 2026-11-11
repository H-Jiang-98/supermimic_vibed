# Runtime configuration

`config.json` selects an environment activation script and the dataset, asset and log directories.
No environment is created or modified by the launchers. The supported environment targets
Isaac Lab commit `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`; your environment must expose
`isaaclab.app.add_launcher_args`, `launch_simulation`, and `isaaclab_newton`.

| Key | Meaning / expected contents |
|---|---|
| `conda_env` | Bash activation script path; shipped value: `tools/activate.sh` |
| `motion_dir` | G1-retargeted LAFAN1 CSVs, 30 fps: root position, root quaternion xyzw, 29 joint angles |
| `robot_description_dir` | `g1_29dof_rev_1_0.urdf`, `g1_29dof.xml`, and `meshes/` |
| `robot_usd_dir` | `g1_29dof/g1_29dof.usda` and its `payloads/` subtree |
| `ground_dir` | `default_ground_plane.usda` and `Materials/Textures/` |
| `logs_dir` | Root for training runs; default `logs`, relative to the config file |

All environment/asset keys above are required. `logs_dir` and the `training` object are optional. Relative paths resolve against the configuration file's directory,
independent of the working directory. Paths support `~`, environment variables, and spaces.
Changing asset directories relocates the existing G1 model; it does not adapt the controller
to a different robot or joint ordering. Keep the mesh and USD payload layouts intact.

`conda_env` is a script path, not an environment name. It supports the same relative/absolute
paths, `~`, environment variables and spaces as the asset paths. For example:

```json
"conda_env": "/path/to/my environment/activate.sh"
```

The script is sourced in Bash without arguments and must activate the desired Python environment
in that shell, returning a nonzero status if activation fails. It needs read permission, not execute
permission. Do not use `exit` on success, since that would terminate the launcher before the task.
The supplied `tools/activate.sh` contains:

```bash
source "${HOME}/miniconda3/etc/profile.d/conda.sh" || return
conda activate supermimic
```

Edit this helper for another Conda installation, or point `conda_env` to your own script.
The launcher does not need `conda` on its initial PATH; it runs `python` from the PATH established
by activation. Shell tools inherit the activated environment too. Activation failures stop execution;
the task's exit code is preserved. The repository working directory is restored after activation.

The launcher chooses `--config`, then `SUPERMIMIC_CONFIG`, then the repository's `config.json`.
Pass `--config` before the script/run name so the launcher can select the environment before
starting Python. The launcher runs scripts from the repository root, so relative output paths
such as `logs/experiment` resolve there.

```bash
./run_train.sh --config /path/to/config.json experiment --iterations 14000
./run_eval.sh --config /path/to/config.json experiment
python3 tools/run.py --config /path/to/config.json tools/test_env.py --num_envs 40
```

For direct execution in an activated environment:

```bash
source tools/activate.sh
python supermimic/train.py --config config.json --log_dir logs/direct --visualizer none
python supermimic/evaluate.py --config config.json --run_dir logs/direct --visualizer none
```

Direct Python execution uses the active interpreter; only `tools/run.py` and the shell wrappers
source `conda_env`. Training/evaluation `--motion_dir` overrides the configured motion directory;
use the same override for both. Training saves the resolved runtime settings in `args.json` and
checkpoint arguments. Evaluation and resuming use the selected runtime config, so paths can
change between machines. Resume does not restore the replay buffer or automatically restore
training flags; repeat the original flags and use `last.pt`.

## Logs and checkpoints

Set `"logs_dir": "/path/to/training logs"` to choose a storage root. Relative paths resolve
against the config file, just like asset paths; omitted `logs_dir` defaults to `logs` there.
The directory is created when training starts.

```bash
./run_train.sh experiment
./run_eval.sh experiment
```

Both wrappers use `<logs_dir>/experiment/`. Training stores `args.json`, `stdout.log`,
`train_log.csv`, `last.pt`, and `ckpt_*.pt` there; evaluation writes `eval.csv` and checkpoint
metrics there. With no run name, both wrappers select `lafan1_main`.
Direct `train.py` / `evaluate.py` calls default to `<logs_dir>/run/`.
Explicit `--log_dir` (training) or `--run_dir` (evaluation) selects a complete run directory
and overrides the configured root. Explicit relative CLI paths resolve from the process working
directory (the repository root when using the launchers). `--resume` and `--ckpt` still accept
explicit checkpoint paths; existing runs are not moved when `logs_dir` changes.

## Training hyperparameters

`config.json` supports JSON with `//` line comments and `/* ... */` block comments (JSONC).
Trailing commas are not supported. Use the project loader rather than `json.load` when reading it.
The shipped `training` object lists every training hyperparameter and annotates its code default.
Omitted fields use their code defaults.

```jsonc
"training": {
  "num_envs": 1024,       // code default: 4096
  "wm_batch": 512,        // code default: 2048
  "pi_batch": 1024,       // code default: 4096
  "iterations": 14000     // code default: 320000
}
```

Precedence: explicit CLI flags > `training` values > defaults in `supermimic/training_config.py`.
Unknown keys, incorrect types, invalid ranges and incompatible buffer/window sizes fail before
training starts. `--use_critic` overrides `"no_critic": true`; `--no_critic` disables the critic.
The final effective training settings (including CLI overrides) are recorded in `args.json` and
checkpoint arguments. Evaluation reads policy architecture and precision from the checkpoint,
not the current `training` settings. `--every N` selects every Nth checkpoint in iteration order,
independent of the configured `ckpt_every` interval.

`--log_dir` overrides in `run_train.sh` move both the metrics/checkpoints and `stdout.log`.
When using wrappers, a misplaced `--config` after the script/run name is rejected to prevent
selecting one environment while loading another configuration's assets.

## World-model training

The default world model follows the official generalized-coordinate implementation:
35 velocity increments, substep-aware pose integration, and a fixed-scale state loss.
See [the equations and input/output conventions](reference.md#world-model).

| Setting | Default | Meaning |
|---|---|---|
| `wm_window` | 8 | Number of state frames; 7 predictions at the default |
| `wm_batches` | 10 | World-model updates per iteration |
| `wm_batch` | 2048 | Windows per update |
| `wm_lr` | 0.0002 | Fixed learning rate for both Muon and Adam |
| `wm_nominal_error_root_pos` | 0.01 | Root-position error scale (m) |
| `wm_nominal_error_root_ori` | 0.02 | Root-rotation error scale (rad) |
| `wm_nominal_error_joint_pos` | 0.1 | Joint-position error scale (rad) |
| `wm_nominal_error_joint_vel` | 3.14 | Joint-velocity error scale (rad/s) |

`wm_window` must be at least 2, and the replay buffer must hold at least `wm_window−1`
transitions. All learning rates and nominal errors must be finite and positive. The WM
loss averages across valid transitions, including terminal transitions and excluding the
rest of a sampled window after a reset. Its scale differs from the previous residual loss.

`lr` and `lr_decay_iters` now control only policy/critic training. `amp` also applies only
to policy/critic MLPs; the world model uses FP32 with TF32 enabled. Muon requires
`torch.optim.Muon` (available in the configured PyTorch 2.11 environment).

New checkpoints use world-model format version 2 and record the integration timestep,
substep count, input-normalization statistics and both optimizer states. Training resume
requires a matching version and integration configuration, plus `last.pt` rather than a
numbered evaluation checkpoint. Old residual-model checkpoints cannot resume this model;
start a fresh run. There is no automatic weight conversion or checkpoint deletion.
Policy-only evaluation still reads its existing observation and architecture metadata.
As before, resume with the original training flags, including the nominal-error settings.

CPU contracts and optional CUDA compilation/update checks:

```bash
source tools/activate.sh
python -m unittest tools.test_world_model tools.test_observation -v
```

The world-model test skips its CUDA check when no GPU is available. For an end-to-end GPU
smoke run (small replay batches, one WM-only iteration, then policy/critic updates):

```bash
./run_train.sh wm_gencoord_smoke --num_envs 32 --buffer_steps 64 \
  --wm_batch 16 --wm_batches 1 --pi_batch 16 --horizon 4 \
  --wm_only_iters 1 --iterations 3 --log_every 1 --ckpt_every 3 --visualizer none
```

## Data and asset conversion

Existing data needs no download or conversion. To download all motions and robot assets,
cache the ground, and generate MJCF/USD into the configured directories:

```bash
python3 tools/run.py tools/download_data.sh
```

This downloads the LAFAN1 G1 retargets and G1 description from
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
LAFAN1 is CC BY-NC-ND 4.0; the Unitree robot description is BSD-3-Clause.
The conversion scripts run as separate Python processes because the importers require offline conversion.
To cache only the ground or rebuild only the robot assets:

```bash
python3 tools/run.py tools/download_ground.sh
python3 tools/run.py tools/urdf_to_mjcf.py
python3 tools/run.py tools/mjcf_to_usd.py
```

The launchers keep Warp, TorchInductor, and Matplotlib caches under `.cache/` unless their cache
environment variables are already set. A C/C++ compiler is needed for `torch.compile`.
Use `CUDA_VISIBLE_DEVICES=0` before a launcher to select the physical GPU; training uses `cuda:0`
within the visible device set. Avoid a virtual-memory `ulimit -v` because CUDA/Warp reserve large
virtual address ranges.

## Policy observation modes

Set `training.obs_mode` in `config.json`; the code and shipped config default to `paper`.
An explicit training `--obs_mode` overrides the config:

```jsonc
"training": {
  "obs_mode": "se23" // paper, so3, se3, se23
}
```

```bash
./run_train.sh pose_tracking --obs_mode se3
./run_train.sh extended_tracking --obs_mode se23
```

| Mode | Appended reference error | Single-frame dimension |
|---|---|---|
| `paper` | None | 93 |
| `so3` | Rotation logarithm φ (3) | 96 |
| `se3` | SE(3) logarithm [φ, ρ] (6) | 99 |
| `se23` | SE₂(3) logarithm [φ, ρ, ν] (9) | 102 |

All modes retain the same 93 proprioceptive values in the existing order: joint position
relative to the default/calibration offset (29), joint velocity (29), body-frame angular
velocity (3), projected gravity (3), and previous action (29). The independent future-goal
input and the world model, critic, reward and termination definitions are unchanged.

For current pelvis state `(R, p, v)` and reference state at frame `f`, the errors are:

```text
φ = Log(Rᵀ R_ref)
ρ = Jₗ(φ)⁻¹ Rᵀ (p_ref − p)
ν = Jₗ(φ)⁻¹ Rᵀ (v_ref − v)
```

`v` is pelvis linear velocity, not angular velocity. Both states use the same world axes;
errors are expressed relative to the current body frame. Jₗ is the SO(3) left Jacobian.
The position and velocity coordinates use its inverse, giving strict group logarithms,
not a concatenation of uncorrected differences. Units are radians, metres and metres/second.
At exactly π the rotation logarithm selects a deterministic branch; it is not continuous
across the branch cut.

The base 93 values keep their existing sensor noise. **Every appended error coordinate has
zero added noise**, including rotation in `so3`. Policy inputs continue to use the learned
running normalization; no additional fixed scaling is applied.

The effective mode is saved in `args.json` and checkpoint arguments. Evaluation always uses
the checkpoint's mode, regardless of the current config. Resume requires the configured mode
to match the checkpoint; repeat `--obs_mode` or use the original config. Dimension mismatches
are rejected before loading policy weights.

CPU regression tests (inside the configured environment):

```bash
python -m unittest tools.test_observation tools.test_motion_goal tools.test_reset_sampling
```

CUDA/Inductor observation forward/backward and float32 boundary checks require a visible GPU:

```bash
python -m unittest tools.test_observation_cuda -v
```

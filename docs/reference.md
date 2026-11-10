# SuperMimic reproduction on Isaac Lab 3 (kit-less, MuJoCo-Warp)

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](../LICENSE).

## Implementation (paper section -> file)

* G1 29-DoF, BeyondMimic actuators (10 Hz natural frequency, damping ratio 2, motor armatures,
  action scale 0.25·τmax/Kp) — `supermimic/robot.py`.
* Reference library: 30 -> 50 Hz resampling, finite-difference velocities, FK of 14 tracked bodies,
  1 s bins — `supermimic/motion.py`, differentiable torch FK `supermimic/kinematics.py`
  (matches the simulator's link poses/velocities to 1e-6).
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), K=10 future reference frames as the goal, BeyondMimic tracking reward without the
  self-collision term and joint-limit weight −2, termination |e_z|>0.25 m or e_tilt>0.8, sim context
  ξ = (friction U[0.3,1.2], joint offsets U[±0.01], torso CoM offset) — `supermimic/mdp.py`.
* Simulator wrapper with return-based reset sampling (50% uniform / 40% softmax(−25 G_b/G_max) /
  10% past-softmax) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on XY/yaw-invariant state features + action + ξ, predicting
  body-frame pose residuals over explicit integration and velocity/joint increments;
  fit on k=8-step autoregressive windows (Eq. 1); analytic reward, observation and termination
  heads — `supermimic/networks.py`, `supermimic/train.py`.
* Policy: motion encoder MLP(2048,1024,512,256)->z∈R^32 (tanh), shared decoder
  MLP(2048,1024,512,256); critic MLP(256,128). BPTT over H=32 imagined steps from replay-buffer
  states with a soft survival weight and critic bootstrap (Eq. 2); critic on GAE(λ=0.95) targets
  (Eq. 3), 5 epochs × 4 minibatches.
* Hyperparameters as Table A.5: 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4 (2.5 vectorized sim steps per iteration), 10 WM batches of 2048 windows per iteration,
  4096 imagined rollouts, Adam lr 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03× over 250k
  iterations, grad clip 1.0 (+ element-wise 0.1 for the policy), γ=0.99.
* Evaluation (App. F.1): each clip once from its first frame, friction 1.0, no DR/noise,
  deterministic policy; failure = training termination — `supermimic/evaluate.py`.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the URDF's
  convex mesh hulls and foot spheres.
* **bf16 autocast for MLP matmuls** (paper: TF32); state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* Policy observation is configurable: `paper` (93), `so3` (96), `se3` (99), or `se23` (102).
  The default is `paper`; appended Lie-algebra errors receive no added noise. See
  [observation definitions](configuration.md#policy-observation-modes).
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up (BeyondMimic's
  start-up events); joint offsets are resampled every episode. Episodes time out after 10 s or at
  the clip end (bootstrapped, not terminal).
* Do not run under a virtual-memory `ulimit -v`: Isaac Lab + Warp reserve ~430 GB of *virtual*
  address space (CUDA VA reservations; RSS ~3 GB) and Warp's CUDA-graph capture needs its mempool,
  which cannot initialize under such a limit. Pin CPUs with `taskset`
  (`CPUS=...` in `run_train.sh`).
* Asset import: the kit-less URDF/MJCF importers fail inside a running Isaac Lab process
  (`PhysxJointAPI` schema not registered), so the USD is converted once, offline, by
  `tools/mjcf_to_usd.py` (registers the importer's schema plugin first) and spawned with
  `UsdFileCfg(variants={"Physics": "mujoco"})`.

## World model

`wm.step(S, a, xi_n)` predicts the next state at **dt = 0.02 s (50 Hz)**.
The state is `S = (p, R, v, ω, q, q̇)`: pelvis position `p` relative to the environment
origin in world axes; body-to-world rotation matrix `R`; world-frame pelvis linear/angular
velocities `v, ω`; and 29 joint positions/velocities `q, q̇`.
For batch size `B`, the shapes are `(B,3)`, `(B,3,3)`, `(B,3)`, `(B,3)`, `(B,29)`, `(B,29)`.
Replay storage replaces `R` with a `wxyz` quaternion, giving **71 values per state**.

**Network input `(B,130)`**, standardized by running mean/std:

- State features **68**: `[p_z, gᴮ, Rᵀv, Rᵀω, q, q̇]`, with dimensions `1+3+3+3+29+29`.
  `gᴮ = Rᵀ(0,0,−9.81)` is body-frame gravity in m/s². These features are invariant to
  horizontal translation and global yaw, and are shared with the critic.
- Action **29**: scaled joint-position target commands `a`.
- Context **33**: normalized `xi_n` containing friction (1), joint zero offsets (29),
  and torso center-of-mass offset (3).

**Network output `(B,70)`**, after denormalization:
`[rₚ, rθ, Δvᴮ, Δωᴮ, Δq, Δq̇]`, with dimensions `3+3+3+3+29+29`.
The pose residuals `rₚ, rθ` are expressed in the current body frame and correct explicit
integration; superscript `B` denotes the same frame. `wm.step()` returns the next state tuple using:

```text
p′ = p + dt v + R rₚ       R′ = R Exp(dt Rᵀω + rθ)
v′ = v + R Δvᴮ             ω′ = ω + R Δωᴮ
q′ = q + Δq                q̇′ = q̇ + Δq̇
```

`Exp` maps a rotation vector to a rotation matrix. Training minimizes multi-step state
prediction error; `targets()` supplies residuals/increments for output-normalization statistics.
See [state features](../supermimic/mdp.py) and [dynamics](../supermimic/networks.py).

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp (Apache-2.0) are used unmodified.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets and description by Unitree
  (BSD-3-Clause). The motion data is not redistributed here and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](../LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.

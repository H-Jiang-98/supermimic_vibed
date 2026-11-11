"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .mdp import INV_DIM, NUM_J, XI_DIM
from .rotations import so3_exp


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


class WorldModelNorm(nn.Module):
    """Explicit empirical normalization: (x - mean) / (std + eps).

    Only freshly collected data updates these statistics. Buffers retain their
    storage for compiled consumers; unlike the policy normalizer, there is no
    sample-count limit. This matches the official dynamics normalizer's math.
    """

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).to(self.mean.dtype)
        n = x.shape[0]
        if n == 0:
            return
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.std.copy_(self.var.sqrt())
        self.count.add_(n)

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


WM_OUT = 6 + NUM_J

# Optional bf16 for policy/critic MLPs. World-model dynamics stay fp32/TF32.
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Official gencoord dynamics, adapted to the existing matrix/world-velocity state.

    Predict [delta_v_body, delta_w_body, delta_joint_velocity], then integrate
    poses analytically with the simulator's substep coefficient. Actions are raw
    policy outputs and xi is the RAW 33-channel simulator context at every call.
    """

    def __init__(self, hidden=(2048, 2048, 2048), *, action_scale, action_offset, dt, num_substeps):
        super().__init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("World-model dt must be finite and positive")
        if type(num_substeps) is not int or num_substeps < 1:
            raise ValueError("World-model num_substeps must be a positive integer")
        if action_scale.shape != (NUM_J,) or action_offset.shape != (NUM_J,):
            raise ValueError(f"World-model action scale/offset must have shape ({NUM_J},)")
        self.dt = float(dt)
        self.num_substeps = num_substeps
        self.substep_coeff = (num_substeps + 1) / (2 * num_substeps)
        self.register_buffer("action_scale", action_scale.detach().float().clone())
        self.register_buffer("action_offset", action_offset.detach().float().clone())
        self.register_buffer("context_mean", torch.tensor([0.75] + [0.] * 32))
        self.register_buffer("context_scale", torch.tensor([0.45] + [0.01] * 29 + [0.025, 0.05, 0.05]))
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.GELU, layer_norm=True)
        self.state_normalizer = WorldModelNorm(INV_DIM)
        self.action_normalizer = WorldModelNorm(NUM_J)

    @staticmethod
    def state_features(S):
        """68 XY/yaw-invariant channels; gravity is a unit direction."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        return torch.cat((p[..., 2:3], -R[..., 2, :], (Rt @ v[..., None])[..., 0],
                          (Rt @ w[..., None])[..., 0], q, qd), -1)

    def proc(self, a):
        return a * self.action_scale + self.action_offset

    @torch.no_grad()
    def update_normalization(self, S, a):
        self.state_normalizer.update(self.state_features(S))
        self.action_normalizer.update(self.proc(a))

    def inputs(self, S, a, xi):
        return torch.cat((self.state_normalizer(self.state_features(S)),
                          self.action_normalizer(self.proc(a)),
                          (xi - self.context_mean) / self.context_scale), -1)

    def apply_prediction(self, S, prediction):
        """Velocity deltas use the current body frame; state velocities use world axes."""
        p, R, v, w, q, qd = S
        dv_b, dw_b, dqd = prediction.split((3, 3, NUM_J), -1)
        dv = (R @ dv_b[..., None])[..., 0]
        dw = (R @ dw_b[..., None])[..., 0]
        w_b = (R.transpose(-1, -2) @ w[..., None])[..., 0]
        c, dt = self.substep_coeff, self.dt
        p1 = p + dt * (v + c * dv)
        R1 = R @ so3_exp(dt * (w_b + c * dw_b))
        return p1, R1, v + dv, w + dw, q + dt * (qd + c * dqd), qd + dqd

    def step(self, S, a, xi):
        # Deliberately bypass run_mlp: AMP_DTYPE only controls policy/critic.
        return self.apply_prediction(S, self.net(self.inputs(S, a, xi)))


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)

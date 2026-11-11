"""Generalized-coordinate world-model fitting, independent of simulator startup.

Semantics follow the official SuperMimic gencoord learner (5efbf11), with
termination masks for this repository's transition-based replay buffer.
"""

import math

import torch
from torch import nn

from .mdp import unflatten_state


WM_LOSS_GROUPS = ("root_pos", "root_ori", "joint_pos", "joint_vel")


def sample_wm_windows(buffer, batch_size, window):
    """A window of L state frames comprises L-1 stored transitions."""
    if window < 2:
        raise ValueError("wm_window must contain at least 2 state frames")
    return buffer.sample_windows(batch_size, window - 1)


def rollout_world_model(wm, states, actions, contexts):
    S = unflatten_state(states[:, 0])
    predictions = []
    for j in range(actions.shape[1]):
        S = wm.step(S, actions[:, j], contexts[:, j])
        predictions.append(S)
    return tuple(torch.stack([state[i] for state in predictions], 1) for i in range(6))


class WorldModelLoss(nn.Module):
    """Bryson-weighted error on root pose and joint state (70 channels).

    Returns (weighted state loss, weighted last-step loss, unweighted group MSE).
    All means exclude transitions after an episode boundary, and use FP32.
    """

    def __init__(self, root_pos=0.01, root_ori=0.02, joint_pos=0.1, joint_vel=3.14):
        super().__init__()
        nominal = (root_pos, root_ori, joint_pos, joint_vel)
        if any(not math.isfinite(x) or x <= 0 for x in nominal):
            raise ValueError("World-model nominal errors must be finite and positive")
        self.widths = (3, 9, 29, 29)
        self.register_buffer("nominal", torch.cat([torch.full((n,), float(x)) for n, x in zip(self.widths, nominal)]))

    @staticmethod
    def blocks(S):
        p, R, _, _, q, qd = S
        return p.float(), R.float().flatten(-2) * math.sqrt(0.5), q.float(), qd.float()

    def forward(self, predicted, target, mask):
        valid = mask.bool()
        # Mask before squaring so excluded, arbitrarily large errors cannot overflow.
        errors = [torch.where(valid[..., None], pred - truth, 0.)
                  for pred, truth in zip(self.blocks(predicted), self.blocks(target))]
        squared = torch.cat(errors, -1).square()
        per_step = (squared / self.nominal.square()).mean(-1)
        count = valid.sum().clamp(min=1)
        loss = per_step.sum() / count
        last = per_step[:, -1].sum() / valid[:, -1].sum().clamp(min=1)
        groups = torch.stack([block.sum() / (count * width)
                              for block, width in zip(squared.split(self.widths, -1), self.widths)])
        return loss, last, groups


def build_wm_optimizers(wm, lr):
    """Official partition: middle Linear matrices to Muon, all other params to Adam."""
    linear = [layer for layer in wm.net if isinstance(layer, nn.Linear)]
    hidden = [layer.weight for layer in linear[1:-1]]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for p in wm.parameters() if id(p) not in hidden_ids]
    device = next(wm.parameters()).device
    cuda = device.type == "cuda"
    optimizers = {"adam": torch.optim.Adam(rest, lr=torch.tensor(lr, device=device) if cuda else lr,
                                         fused=cuda, capturable=cuda)}
    if hidden:
        optimizers["muon"] = torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")
    return optimizers


def load_wm_optimizer_state(optimizers, states):
    if set(states) != set(optimizers):
        raise ValueError("World-model optimizer groups do not match the checkpoint")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(states[name])
        # map_location='cpu' also moves saved scalar learning-rate tensors.
        # Optimizer.load_state_dict only relocates the per-parameter state.
        for group in optimizer.param_groups:
            if torch.is_tensor(group["lr"]):
                group["lr"] = group["lr"].to(group["params"][0].device)


def optimize_world_model(wm, optimizers, loss):
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    for optimizer in optimizers.values():
        optimizer.step()
    return norm.detach()

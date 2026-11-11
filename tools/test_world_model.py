"""Gencoord dynamics, loss, training-window and checkpoint contracts (CPU/CUDA)."""

import copy
import io
import math
import unittest

import torch
from torch import nn

from supermimic.buffer import FIELDS, ReplayBuffer
from supermimic.checkpoints import WM_FORMAT_VERSION, checkpoint_world_model
from supermimic.config import load_config
from supermimic.mdp import flatten_state
import supermimic.networks as networks
from supermimic.networks import WorldModel, WorldModelNorm
from supermimic.rotations import skew, so3_exp
from supermimic.training_config import TRAIN_DEFAULTS, validate_training
from supermimic.wm_training import (WorldModelLoss, build_wm_optimizers, load_wm_optimizer_state,
                                   optimize_world_model, rollout_world_model, sample_wm_windows)


def model(n=4, dtype=torch.float32, device="cpu"):
    return WorldModel(hidden=(16, 16, 16), action_scale=torch.linspace(.1, .5, 29),
                      action_offset=torch.linspace(-.3, .3, 29), dt=.02, num_substeps=n).to(device=device, dtype=dtype)


def state(*shape, dtype=torch.float32, device="cpu"):
    rand = lambda n: torch.randn(*shape, n, dtype=dtype, device=device) * .2
    return rand(3), so3_exp(rand(3)), rand(3), rand(3), rand(29), rand(29)


def context(*shape, device="cpu"):
    xi = torch.zeros(*shape, 33, device=device)
    xi[..., 0] = .75
    return xi


class WorldModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(41)

    def test_input_processing_and_normalization(self):
        wm = model()
        S, actions = state(2, 3), torch.randn(2, 3, 29)
        xi = context(2, 3)
        xi[0] = torch.tensor([1.2] + [.01] * 29 + [.025, .05, .05])
        xi[1] = torch.tensor([.3] + [-.01] * 29 + [-.025, -.05, -.05])
        pointers = {k: v.data_ptr() for k, v in wm.named_buffers()}
        wm.update_normalization(S, actions)
        features = wm.state_features(S)
        torch.testing.assert_close(features[..., 1:4].norm(dim=-1), torch.ones(2, 3))
        torch.testing.assert_close(wm.state_normalizer.mean, features.flatten(0, 1).mean(0))
        processed = actions * wm.action_scale + wm.action_offset
        torch.testing.assert_close(wm.action_normalizer.mean, processed.flatten(0, 1).mean(0))
        inputs = wm.inputs(S, actions, xi)
        self.assertEqual(inputs.shape, (2, 3, 130))
        torch.testing.assert_close(inputs[..., 68:97], (processed - processed.mean((0, 1))) /
                                   (processed.flatten(0, 1).std(0, unbiased=False) + .01))
        torch.testing.assert_close(inputs[0, :, 97:], torch.ones(3, 33))
        torch.testing.assert_close(inputs[1, :, 97:], -torch.ones(3, 33))
        before = copy.deepcopy(wm.state_dict())
        wm.train()
        wm.step(S, actions, xi)
        wm.eval()
        wm.step(S, actions, xi)
        for key, value in wm.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(pointers, {k: v.data_ptr() for k, v in wm.named_buffers()})
        self.assertTrue(all(isinstance(wm.net[i], nn.GELU) for i in (2, 5, 8)))
        self.assertEqual(wm.net[-1].out_features, 35)
        self.assertGreater(wm.net[-1].weight.abs().sum(), 0)

    def test_population_stats_and_no_count_limit(self):
        norm = WorldModelNorm(4).double()
        x = torch.randn(37, 4, dtype=torch.float64)
        for batch in (x[:1], x[1:12], x[12:]):
            norm.update(batch)
        torch.testing.assert_close(norm.mean, x.mean(0))
        torch.testing.assert_close(norm.var, x.var(0, unbiased=False))
        torch.testing.assert_close(norm(x), (x - x.mean(0)) / (x.std(0, unbiased=False) + .01))
        norm.count.fill_(200_000_000)
        norm.update(x[:1])
        self.assertEqual(norm.count.item(), 200_000_001)
        constant = WorldModelNorm(4)
        constant.update(torch.ones(1, 4))
        torch.testing.assert_close(constant(torch.ones(1, 4)), torch.zeros(1, 4))

    def test_integration_matches_substeps_and_matrix_exponential(self):
        # Independent constant-acceleration substep integration in world axes.
        for n in (1, 2, 4):
            for shape in ((3,), (2, 3)):
                wm = model(n, torch.float64)
                S = state(*shape, dtype=torch.float64)
                p, R, v, w, q, qd = S
                delta = torch.randn(*shape, 35, dtype=torch.float64) * .2
                dv, dw = (R @ delta[..., :3, None])[..., 0], (R @ delta[..., 3:6, None])[..., 0]
                position, joints = p.clone(), q.clone()
                for j in range(1, n + 1):
                    position += wm.dt / n * (v + j / n * dv)
                    joints += wm.dt / n * (qd + j / n * delta[..., 6:])
                expected_R = torch.matrix_exp(skew(wm.dt * (w + (n + 1) / (2 * n) * dw))) @ R
                expected = (position, expected_R, v + dv, w + dw, joints, qd + delta[..., 6:])
                for actual, reference in zip(wm.apply_prediction(S, delta), expected):
                    torch.testing.assert_close(actual, reference, atol=2e-11, rtol=2e-11)
                zero = wm.apply_prediction(S, torch.zeros_like(delta))
                torch.testing.assert_close(zero[0], p + wm.dt * v)
                torch.testing.assert_close(zero[4], q + wm.dt * qd)

    def test_invariant_features_and_equivariant_step(self):
        wm = model(dtype=torch.float64)
        S = state(4, dtype=torch.float64)
        yaw = so3_exp(torch.tensor([0., 0., 1.3], dtype=torch.float64))
        shift = torch.tensor([12., -3., 0.], dtype=torch.float64)
        rot = lambda x: (yaw @ x[..., None])[..., 0]
        transform = lambda s: (rot(s[0]) + shift, yaw @ s[1], rot(s[2]), rot(s[3]), s[4], s[5])
        a = torch.randn(4, 29, dtype=torch.float64)
        xi = context(4).double()
        torch.testing.assert_close(wm.state_features(S), wm.state_features(transform(S)))
        for x, y in zip(wm.step(transform(S), a, xi), transform(wm.step(S, a, xi))):
            torch.testing.assert_close(x, y)

    def test_prediction_gradient_at_zero_and_multistep_actions(self):
        wm = model(dtype=torch.float64)
        S = state(2, dtype=torch.float64)
        delta = torch.zeros(2, 35, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda d: wm.apply_prediction(S, d), (delta,)))
        wm = model()
        wm.requires_grad_(False)
        actions = (torch.randn(2, 32, 29) * .1).requires_grad_()
        current = state(2)
        for j in range(32):
            current = wm.step(current, actions[:, j], context(2))
        sum(x.square().sum() for x in current).backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertTrue((actions.grad.abs().sum((0, 2)) > 0).all())
        R = current[1]
        torch.testing.assert_close(R.transpose(-1, -2) @ R, torch.eye(3).expand(2, 3, 3), atol=3e-6, rtol=3e-6)
        features = wm.state_features(current)
        torch.testing.assert_close(features[..., :1], current[0][..., 2:3])
        torch.testing.assert_close(features[..., 1:4], -R[..., 2, :])

    def test_loss_channels_weights_and_masks(self):
        loss_fn = WorldModelLoss()
        truth = state(2, 3)
        pred = list(copy.deepcopy(truth))
        pred[0] += .01
        pred[1] = so3_exp(torch.tensor([.02, 0., 0.])) @ pred[1]
        pred[2] += 100  # Root twist is not independently scored.
        pred[3] += 100
        pred[4] += .1
        pred[5] += 3.14
        mask = torch.tensor([[1., 1., 1.], [1., 0., 0.]])
        mask_bool = mask.bool()
        rot_sq = ((pred[1] - truth[1]).square() * .5).sum(-1).sum(-1)
        expected_step = (3 + rot_sq / .02**2 + 29 + 29) / 70
        loss, last, groups = loss_fn(tuple(pred), truth, mask)
        torch.testing.assert_close(loss, expected_step[mask_bool].mean())
        torch.testing.assert_close(last, expected_step[0, -1])
        torch.testing.assert_close(groups, torch.stack([torch.tensor(.01**2), rot_sq[mask_bool].mean() / 9,
                                                      torch.tensor(.1**2), torch.tensor(3.14**2)]))
        for part in pred:
            part[1, 1:] = 1e30
        masked = tuple(x.detach().requires_grad_() for x in pred)
        actual = loss_fn(masked, truth, mask)[0]
        torch.testing.assert_close(actual, loss)
        actual.backward()
        for i, part in enumerate(masked):
            if i in (2, 3):
                self.assertIsNone(part.grad)
            else:
                self.assertTrue(torch.isfinite(part.grad).all())
                self.assertEqual(part.grad[1, 1:].abs().sum(), 0)
        for item in loss_fn(tuple(pred), truth, torch.zeros_like(mask)):
            torch.testing.assert_close(item, torch.zeros_like(item))

    def test_window_frame_count_and_terminal_transition(self):
        buffer = ReplayBuffer(7, 1, "cpu")
        for j in range(7):
            tr = {key: torch.zeros((1,) + ((shape,) if isinstance(shape, int) else shape), dtype=dtype)
                  for key, (shape, dtype) in FIELDS.items()}
            tr["s"] = flatten_state(state(1))
            tr["s1"] = tr["s"].clone()
            tr["xi"] = context(1)
            tr["done"].fill_(j == 2)
            buffer.add(tr)
        window = sample_wm_windows(buffer, 2, 8)
        self.assertEqual(window["a"].shape, (2, 7, 29))
        torch.testing.assert_close(window["mask"], torch.tensor([[1., 1., 1., 0., 0., 0., 0.]]).expand(2, 7))
        wm = model()
        pred = rollout_world_model(wm, window["s"], window["a"], window["xi"])
        self.assertEqual(pred[0].shape, (2, 7, 3))
        with self.assertRaisesRegex(ValueError, "2 state frames"):
            sample_wm_windows(buffer, 2, 1)
        self.assertEqual(sample_wm_windows(buffer, 2, 2)["a"].shape[1], 1)

    def test_loss_reaches_every_velocity_output(self):
        wm = model()
        S = state(2, 3)
        target = wm.apply_prediction(S, torch.zeros(2, 3, 35))
        delta = torch.full((2, 3, 35), .1, requires_grad=True)
        loss = WorldModelLoss()(wm.apply_prediction(S, delta), target, torch.ones(2, 3))[0]
        loss.backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertTrue((delta.grad.abs().sum((0, 1)) > 0).all())

    def test_optimizer_partition_training_and_checkpoint(self):
        wm = model()
        opts = build_wm_optimizers(wm, .002)
        ids = {name: [id(p) for group in opt.param_groups for p in group["params"]] for name, opt in opts.items()}
        linear = [m for m in wm.net if isinstance(m, nn.Linear)]
        self.assertEqual(set(ids["muon"]), {id(linear[1].weight), id(linear[2].weight)})
        self.assertFalse(set(ids["muon"]) & set(ids["adam"]))
        self.assertEqual(set(ids["muon"] + ids["adam"]), {id(p) for p in wm.parameters()})
        self.assertEqual(len(ids["muon"] + ids["adam"]), len(list(wm.parameters())))
        loss_fn = WorldModelLoss()
        S = state(8)
        # A constant-velocity target has a consistent pose update.
        target = wm.apply_prediction(S, torch.zeros(8, 35))
        a, xi = torch.randn(8, 29), context(8)
        wm.update_normalization(S, a)
        def objective(net):
            pred = tuple(x[:, None] for x in net.step(S, a, xi))
            return loss_fn(pred, tuple(x[:, None] for x in target), torch.ones(8, 1))[0]
        initial = objective(wm).item()
        for _ in range(25):
            optimize_world_model(wm, opts, objective(wm))
        self.assertLess(objective(wm).item(), initial * .5)
        ck = {"wm": wm.state_dict(), "opt_wm": {name: opt.state_dict() for name, opt in opts.items()},
              "wm_format_version": WM_FORMAT_VERSION, "wm_config": {"dt": .02, "num_substeps": 4}}
        stream = io.BytesIO()
        torch.save(ck, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True, map_location="cpu")
        checkpoint_world_model(saved, dt=.02, num_substeps=4)
        restored = model()
        restored.load_state_dict(saved["wm"])
        restored_opts = build_wm_optimizers(restored, .002)
        load_wm_optimizer_state(restored_opts, saved["opt_wm"])
        optimize_world_model(wm, opts, objective(wm))
        optimize_world_model(restored, restored_opts, objective(restored))
        for name, value in wm.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        for invalid in ({}, {**saved, "wm_format_version": 1}):
            with self.assertRaisesRegex(ValueError, "start a new training run"):
                checkpoint_world_model(invalid, dt=.02, num_substeps=4)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            checkpoint_world_model(saved, dt=.02, num_substeps=2)
        with self.assertRaisesRegex(ValueError, "last.pt"):
            checkpoint_world_model({**saved, "opt_wm": {}}, dt=.02, num_substeps=4)

    def test_configuration(self):
        config = load_config()["training"]
        for key in TRAIN_DEFAULTS:
            if key.startswith("wm_"):
                self.assertEqual(config[key], TRAIN_DEFAULTS[key])
        validate_training({"buffer_steps": 7, "wm_window": 8})
        validate_training({"buffer_steps": 1, "wm_window": 2})
        for key in ("wm_lr", "wm_nominal_error_root_pos", "wm_nominal_error_root_ori",
                    "wm_nominal_error_joint_pos", "wm_nominal_error_joint_vel"):
            for value in (0, -1, math.inf, math.nan, True):
                with self.assertRaises(ValueError):
                    validate_training({key: value})
        for invalid in ({"wm_window": 1}, {"buffer_steps": 6, "wm_window": 8}):
            with self.assertRaises(ValueError):
                validate_training(invalid)

    def test_compiled_cpu_and_amp_isolation(self):
        wm = model()
        S, a, xi = state(2), torch.randn(2, 29), context(2)
        expected = wm.step(S, a, xi)
        old_amp = networks.AMP_DTYPE
        try:
            networks.AMP_DTYPE = torch.bfloat16
            for x, y in zip(expected, wm.step(S, a, xi)):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        finally:
            networks.AMP_DTYPE = old_amp
        compiled = torch.compile(wm.step, backend="aot_eager", fullgraph=True)
        ac = a.detach().requires_grad_()
        ae = a.detach().requires_grad_()
        actual, expected = compiled(S, ac, xi), wm.step(S, ae, xi)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y)
        sum(x.square().sum() for x in actual).backward()
        sum(x.square().sum() for x in expected).backward()
        torch.testing.assert_close(ac.grad, ae.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_compiled_update_and_resume(self):
        # Match training's TF32 setting without affecting the other tests.
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        wm = model(device="cuda")
        opts = build_wm_optimizers(wm, .0002)
        S = state(4, 7, device="cuda")
        states = flatten_state(S)
        actions = torch.randn(4, 7, 29, device="cuda")
        xi = context(4, 7, device="cuda")
        wm.update_normalization(S, actions)
        loss_fn = WorldModelLoss().cuda()
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], device="cuda")
        rollout = torch.compile(lambda s, a, c: rollout_world_model(wm, s, a, c), fullgraph=True)
        loss_c = torch.compile(loss_fn, fullgraph=True)
        eager = rollout_world_model(wm, states, actions, xi)
        compiled = rollout(states, actions, xi)
        for actual, expected in zip(compiled, eager):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        eager_loss = loss_fn(eager, S, mask)[0]
        compiled_loss = loss_c(compiled, S, mask)[0]
        torch.testing.assert_close(compiled_loss, eager_loss, atol=2e-5, rtol=2e-4)
        eager_grads = torch.autograd.grad(eager_loss, tuple(wm.parameters()))
        compiled_grads = torch.autograd.grad(compiled_loss, tuple(wm.parameters()))
        for actual, expected in zip(compiled_grads, eager_grads):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
        for _ in range(3):
            loss = loss_c(rollout(states, actions, xi), S, mask)[0]
            self.assertTrue(torch.isfinite(optimize_world_model(wm, opts, loss)))
        stream = io.BytesIO()
        torch.save({"wm": wm.state_dict(), "opt_wm": {k: opt.state_dict() for k, opt in opts.items()}}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = model(device="cuda")
        restored.load_state_dict(saved["wm"])
        restored_opts = build_wm_optimizers(restored, .0002)
        load_wm_optimizer_state(restored_opts, saved["opt_wm"])
        for group in restored_opts["adam"].param_groups:
            self.assertEqual(group["lr"].device.type, "cuda")
        # Identical next update after a real CPU -> CUDA model/optimizer restore.
        for net, optimizers in ((wm, opts), (restored, restored_opts)):
            loss = loss_fn(rollout_world_model(net, states, actions, xi), S, mask)[0]
            self.assertTrue(torch.isfinite(optimize_world_model(net, optimizers, loss)))
        for name, value in wm.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        loss = loss_c(rollout(states, actions, xi), S, mask)[0]
        self.assertTrue(torch.isfinite(optimize_world_model(wm, opts, loss)))


if __name__ == "__main__":
    unittest.main()

"""Observation modes, Lie logarithms, and policy/checkpoint integration (CPU)."""
import argparse
import io
import math
from types import SimpleNamespace
import unittest

import torch

from supermimic.checkpoints import checkpoint_obs_mode
from supermimic.config import load_config
from supermimic.mdp import MDP, BASE_OBS_NOISE
from supermimic.networks import Policy, WorldModel
from supermimic.rotations import so3_exp, se3_log, se23_log, skew
from supermimic.training_config import OBS_DIMS, add_training_args, validate_training


class ObservationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.dtype = torch.float64
        self.kin = SimpleNamespace(q_lo=-torch.ones(29), q_hi=torch.ones(29))
        self.motion = SimpleNamespace(root_R=so3_exp(torch.randn(6, 3, dtype=self.dtype) * .3),
                                      root_p=torch.randn(6, 3, dtype=self.dtype),
                                      root_v=torch.randn(6, 3, dtype=self.dtype))

    def mdp(self, mode):
        return MDP(self.kin, self.motion, torch.zeros(29, dtype=self.dtype), obs_mode=mode)

    def state(self, shape):
        rand = lambda n: torch.randn(*shape, n, dtype=self.dtype)
        return rand(3), so3_exp(rand(3) * .3), rand(3), rand(3), rand(29), rand(29)

    def test_shapes_base_and_noise(self):
        for shape in ((2,), (2, 3)):
            S = self.state(shape)
            a, xi = torch.randn(*shape, 29), torch.randn(*shape, 33)
            f = torch.zeros(shape, dtype=torch.long)
            base = self.mdp('paper').obs(S, a, xi, f)
            for mode, dim in OBS_DIMS.items():
                mdp = self.mdp(mode)
                obs = mdp.obs(S, a, xi, f)
                self.assertEqual(obs.shape, (*shape, dim))
                torch.testing.assert_close(obs[..., :93], base)
                noisy = mdp.add_noise(obs)
                torch.testing.assert_close(noisy[..., 93:], obs[..., 93:], rtol=0, atol=0)
                self.assertTrue(((noisy[..., :93] - base).abs() <= BASE_OBS_NOISE + 1e-8).all())
        # Paper observations do not access reference data.
        mdp = self.mdp('paper')
        mdp.motion = None
        torch.testing.assert_close(mdp.obs(S, a, xi, f), base)

    def test_reference_zero_and_translation_velocity(self):
        f = torch.arange(6)
        m = self.motion
        z = torch.zeros(6, 29, dtype=self.dtype)
        xi = torch.zeros(6, 33, dtype=self.dtype)
        S = (m.root_p, m.root_R, m.root_v, torch.zeros(6, 3, dtype=self.dtype), z, z)
        for mode in OBS_DIMS:
            torch.testing.assert_close(self.mdp(mode).obs(S, z, xi, f)[..., 93:],
                                       torch.zeros(6, OBS_DIMS[mode] - 93, dtype=self.dtype), atol=1e-9, rtol=0)
        dp, dv = torch.randn(6, 3, dtype=self.dtype), torch.randn(6, 3, dtype=self.dtype)
        S = (m.root_p - dp, m.root_R, m.root_v - dv, S[3], z, z)
        error = self.mdp('se23').obs(S, z, xi, f)[..., 93:]
        Rt = m.root_R.transpose(-1, -2)
        torch.testing.assert_close(error[..., 3:6], (Rt @ dp[..., None])[..., 0])
        torch.testing.assert_close(error[..., 6:9], (Rt @ dv[..., None])[..., 0])

    def test_matrix_exponential_roundtrip_and_gradients(self):
        axis = torch.tensor([1., 2., -3.], dtype=self.dtype)
        axis = axis / axis.norm()
        for angle in (0., 1e-8, .009, .011, .8, math.pi - 1e-5, math.pi):
            phi = axis * angle
            for dim, log in ((6, se3_log), (9, se23_log)):
                algebra = torch.zeros(4 if dim == 6 else 5, 4 if dim == 6 else 5, dtype=self.dtype)
                algebra[:3, :3] = skew(phi)
                algebra[:3, 3:] = torch.randn(3, algebra.shape[0] - 3, dtype=self.dtype)
                group = torch.matrix_exp(algebra)
                parts = [group[:3, :3], group[:3, 3]]
                if dim == 9:
                    parts.append(group[:3, 4])
                coords = log(*parts)
                recovered = torch.zeros_like(algebra)
                recovered[:3, :3] = skew(coords[:3])
                recovered[:3, 3:] = coords[3:].reshape(-1, 3).T
                torch.testing.assert_close(torch.matrix_exp(recovered), group, atol=2e-9, rtol=2e-9)
            if angle < math.pi:
                x = (axis * angle).requires_grad_()
                p = torch.randn(3, dtype=self.dtype, requires_grad=True)
                v = torch.randn(3, dtype=self.dtype, requires_grad=True)
                fn = lambda x, p, v: se23_log(so3_exp(x), p, v)
                self.assertTrue(torch.autograd.gradcheck(fn, (x, p, v), eps=1e-7, atol=1e-5))

    def test_global_frame_invariance(self):
        S = self.state((6,))
        Q = so3_exp(torch.tensor([.5, -.2, .4], dtype=self.dtype))
        t = torch.tensor([3., -2., 1.], dtype=self.dtype)
        rotate = lambda x: (Q @ x[..., None])[..., 0]
        transformed = (rotate(S[0]) + t, Q @ S[1], rotate(S[2]), rotate(S[3]), S[4], S[5])
        a, xi, f = torch.zeros(6, 29), torch.zeros(6, 33), torch.arange(6)
        for mode in ('so3', 'se3', 'se23'):
            mdp = self.mdp(mode)
            original = mdp.obs(S, a, xi, f)[..., 93:]
            mdp.motion = SimpleNamespace(root_R=Q @ self.motion.root_R,
                                        root_p=rotate(self.motion.root_p) + t,
                                        root_v=rotate(self.motion.root_v))
            torch.testing.assert_close(mdp.obs(transformed, a, xi, f)[..., 93:], original)

    def test_policy_rollout_and_checkpoint(self):
        torch.set_num_threads(2)
        for mode, dim in OBS_DIMS.items():
            mdp = self.mdp(mode)
            pi = Policy(dim, 73, hist=2)
            wm = WorldModel(hidden=(16,), action_scale=torch.ones(29), action_offset=torch.zeros(29),
                            dt=0.02, num_substeps=4)
            S = tuple(x.float() for x in self.state((2,)))
            mdp.default_q = mdp.default_q.float()
            mdp.motion = SimpleNamespace(**{k: v.float() for k, v in vars(self.motion).items()})
            xi, a, f, goal = torch.zeros(2, 33), torch.zeros(2, 29), torch.zeros(2, dtype=torch.long), torch.randn(2, 73)
            xi[:, 0] = .75  # WorldModel.step receives raw simulator context.
            hist = mdp.obs(S, a, xi, f).unsqueeze(1).repeat(1, 2, 1)
            for _ in range(2):
                a = pi(hist, goal)
                S = wm.step(S, a, xi)
                f = f + 1
                hist = torch.cat((hist[:, 1:], mdp.add_noise(mdp.obs(S, a, xi, f)).unsqueeze(1)), 1)
            loss = S[0].square().sum() + pi(hist, goal).square().sum()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in pi.parameters()))
            ck = {'args': {'obs_mode': mode}, 'pi': pi.state_dict()}
            stream = io.BytesIO()
            torch.save(ck, stream)
            stream.seek(0)
            loaded = torch.load(stream, weights_only=True)
            self.assertEqual(checkpoint_obs_mode(loaded, mode), mode)
            pi.load_state_dict(loaded['pi'])
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                checkpoint_obs_mode(loaded, 'paper' if mode != 'paper' else 'so3')
        with self.assertRaisesRegex(ValueError, 'dimension'):
            checkpoint_obs_mode({'args': {'obs_mode': 'so3'}, 'pi': {'obs_norm.mean': torch.zeros(99)}})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            checkpoint_obs_mode({'args': {'obs_mode': 'invalid'}})

    def test_configuration(self):
        self.assertEqual(validate_training({})['obs_mode'], 'paper')
        self.assertEqual(load_config()['training']['obs_mode'], 'paper')
        for mode in OBS_DIMS:
            self.assertEqual(validate_training({'obs_mode': mode})['obs_mode'], mode)
        for invalid in ('invalid', '', 2, None):
            with self.assertRaises(ValueError):
                validate_training({'obs_mode': invalid})
        parser = argparse.ArgumentParser()
        add_training_args(parser, {'obs_mode': 'se3'})
        self.assertEqual(parser.parse_args([]).obs_mode, 'se3')
        self.assertEqual(parser.parse_args(['--obs_mode', 'so3']).obs_mode, 'so3')
        with self.assertRaises(ValueError):
            self.mdp('invalid')


if __name__ == '__main__':
    unittest.main()

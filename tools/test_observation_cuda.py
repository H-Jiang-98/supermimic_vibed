"""CUDA/Inductor checks for the observation head used in imagined rollouts."""
from types import SimpleNamespace
import unittest

import torch

from supermimic.mdp import MDP
from supermimic.rotations import se23_log, skew, so3_exp
from supermimic.training_config import OBS_DIMS


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class ObservationCudaTest(unittest.TestCase):
    def test_compiled_observation_and_gradients(self):
        torch.manual_seed(9)
        device = 'cuda:0'
        rand = lambda *shape: torch.randn(*shape, device=device)
        motion = SimpleNamespace(root_p=rand(8, 3), root_R=so3_exp(rand(8, 3) * .3), root_v=rand(8, 3))
        kin = SimpleNamespace(q_lo=-torch.ones(29, device=device), q_hi=torch.ones(29, device=device))
        # Include the history batch shape used by replay sampling.
        S = (rand(2, 3, 3), so3_exp(rand(2, 3, 3) * .2), rand(2, 3, 3), rand(2, 3, 3),
             rand(2, 3, 29), rand(2, 3, 29))
        a, xi = rand(2, 3, 29), rand(2, 3, 33)
        f = torch.arange(6, device=device).reshape(2, 3)
        for mode, dim in OBS_DIMS.items():
            with self.subTest(mode=mode):
                mdp = MDP(kin, motion, torch.zeros(29, device=device), obs_mode=mode)
                eager_state = tuple(x.detach().requires_grad_() for x in S)
                compiled_state = tuple(x.detach().requires_grad_() for x in S)
                expected = mdp.obs(eager_state, a, xi, f)
                actual = torch.compile(mdp.obs, fullgraph=True)(compiled_state, a, xi, f)
                self.assertEqual(actual.shape, (2, 3, dim))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for eager, compiled in zip(eager_state, compiled_state):
                    if eager.grad is None:
                        self.assertIsNone(compiled.grad)
                    else:
                        self.assertTrue(torch.isfinite(compiled.grad).all())
                        torch.testing.assert_close(compiled.grad, eager.grad, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(mdp.add_noise(actual)[..., 93:], actual[..., 93:], atol=0, rtol=0)

    def test_float32_logarithm_boundaries(self):
        angles = torch.tensor([0., 1e-7, .0099, .0101, .5, torch.pi - 1e-4, torch.pi], device='cuda')
        axis = torch.tensor([1., 2., -3.], device='cuda')
        phi = angles[:, None] * (axis / axis.norm())
        algebra = torch.zeros(7, 5, 5, device='cuda')
        algebra[:, :3, :3] = skew(phi)
        algebra[:, :3, 3:] = torch.randn(7, 3, 2, device='cuda')
        group = torch.matrix_exp(algebra)
        R = group[:, :3, :3].detach().requires_grad_()
        p = group[:, :3, 3].detach().requires_grad_()
        v = group[:, :3, 4].detach().requires_grad_()
        coords = se23_log(R, p, v)
        recovered = torch.zeros_like(algebra)
        recovered[:, :3, :3] = skew(coords[:, :3])
        recovered[:, :3, 3:] = coords[:, 3:].reshape(7, 2, 3).transpose(-1, -2)
        torch.testing.assert_close(torch.matrix_exp(recovered), group, atol=3e-5, rtol=3e-5)
        coords.square().sum().backward()
        for tensor in (R, p, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == '__main__':
    unittest.main()

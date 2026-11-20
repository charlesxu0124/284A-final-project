"""Cross-backend checks for the stock tabletop fold."""

import unittest

import numpy as np
import torch

from benchmark import DT, fold_actions, run
from cloth_sim import Cloth, MassSpring, XPBD
from pgrd import PGRD, VelocityResidual
from torch_sim import TorchMassSpring, TorchXPBD, prepare_actions, rollout


class SimulationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.resolution = 8
        self.batch = 3
        self.cloth = Cloth.square(self.resolution)
        self.corners, self.targets, self.held = fold_actions(self.cloth, self.resolution, 32, self.batch)
        self.cpu = torch.device("cpu")
        self.actions = prepare_actions(self.cloth, self.corners, self.targets, self.held, self.cpu)

    def assert_close_trajectory(self, expected, actual):
        error = np.sqrt(np.mean(np.sum((expected - actual)**2, axis=-1)))
        self.assertLess(error, 0.0002, "trajectory error exceeds 0.2 mm")
        self.assertTrue(np.all(actual[..., 2] >= -1e-6))
        np.testing.assert_allclose(actual[1:][:, :, self.corners[:2]], self.targets[:, :, :2], atol=1e-6)
        np.testing.assert_allclose(actual[1:][self.held][:, :, self.corners[2:]],
                                   self.targets[self.held, :, 2:], atol=1e-6)

    def test_torch_cpu_matches_numpy(self):
        for numpy_cls, torch_cls, settings in ((MassSpring, TorchMassSpring, (12,)),
                                               (XPBD, TorchXPBD, (2, 4))):
            with self.subTest(solver=torch_cls.__name__):
                expected, _ = run(numpy_cls(self.cloth, self.batch, *settings), DT,
                                  self.corners, self.targets, self.held, warmup=0)
                actual, _ = rollout(torch_cls(self.cloth, self.batch, *settings, self.cpu), DT, self.actions)
                self.assert_close_trajectory(expected, actual.numpy())

    def test_zero_hybrid_reproduces_calibrated_spring(self):
        spring = TorchMassSpring(self.cloth, self.batch, 12, self.cpu, friction=0.55)
        expected, _ = rollout(spring, DT, self.actions)
        model = VelocityResidual(self.cloth).eval()
        actual, _ = rollout(PGRD(spring, model), DT, self.actions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_prefix_and_disjoint_trajectories(self):
        def get_targets(split, batch):
            return fold_actions(self.cloth, self.resolution, 32, batch, split, seed=7)[1]

        train, validation, test = [get_targets(split, self.batch) for split in ("train", "validation", "test")]
        self.assertEqual(len({fold.tobytes() for split in (train, validation, test) for fold in split.swapaxes(0, 1)}),
                         3 * self.batch)
        np.testing.assert_array_equal(test, get_targets("test", 2 * self.batch)[:, :self.batch])

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs an Apple Metal GPU")
    def test_mps_matches_torch_cpu(self):
        from metal_sim import MetalMassSpring, MetalXPBD

        mps = torch.device("mps")
        actions = prepare_actions(self.cloth, self.corners, self.targets, self.held, mps)
        for torch_cls, metal_cls, settings in ((TorchMassSpring, MetalMassSpring, (12,)),
                                               (TorchXPBD, MetalXPBD, (2, 4))):
            with self.subTest(solver=metal_cls.__name__):
                expected, _ = rollout(torch_cls(self.cloth, self.batch, *settings, self.cpu), DT, self.actions)
                for solver_class in (torch_cls, metal_cls):
                    actual, _ = rollout(solver_class(self.cloth, self.batch, *settings, mps), DT, actions)
                    self.assert_close_trajectory(expected.numpy(), actual.cpu().numpy())


if __name__ == "__main__":
    unittest.main()

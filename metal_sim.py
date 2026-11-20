"""Fused Metal physics for regular cloth meshes on Apple GPUs."""

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import torch

from cloth_sim import edge_color_groups
from torch_sim import TorchSolver


@lru_cache(maxsize=4)
def load_kernels(capacity):
    source = Path(__file__).with_name("cloth_kernels.metal").read_text(encoding="utf-8")
    return torch.mps.compile_shader(f"#define VERTEX_CAPACITY {capacity}\n" + source)


class MetalSolver(TorchSolver):
    def __init__(self, cloth, batch, substeps, device, friction):
        if device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
            raise ValueError("fused Metal physics requires an MPS device and torch.mps.compile_shader")
        if len(cloth.rest) > 1024:
            raise ValueError("fused Metal physics supports up to 1024 vertices per cloth")
        super().__init__(cloth, batch, substeps, device, friction)
        self.group_size = math.ceil(len(cloth.rest) / 32) * 32
        self.library = load_kernels(self.group_size)
        self.physical = self.rest.new_tensor([self.inverse_mass, friction])

    def reset(self):
        super().reset()
        self.positions = self.positions.contiguous()

    def integers(self, values):
        return torch.as_tensor(values, device=self.device, dtype=torch.int32).contiguous()

    def step(self, dt, action):
        h = dt / self.substeps
        next_positions = torch.empty_like(self.positions, memory_format=torch.contiguous_format)
        next_velocities = torch.empty_like(next_positions)
        self.kernel(self.positions.contiguous(), self.velocities.contiguous(), next_positions, next_velocities,
                    action.targets, action.slots, self.offsets, self.connectivity, self.material,
                    self.config, self.physical, h, math.exp(-2 * h), action.indices.numel(),
                    threads=self.batch * self.group_size, group_size=self.group_size)
        self.positions, self.velocities = next_positions, next_velocities
        return next_positions


class MetalMassSpring(MetalSolver):
    def __init__(self, cloth, batch, substeps, device, stiffness_scale=1.0, friction=0.4):
        super().__init__(cloth, batch, substeps, device, friction)
        self.kernel = self.library.spring_step
        incidence = [[] for _ in cloth.rest]
        for edge, (left, right) in enumerate(cloth.edges):
            incidence[left].append((right, edge))
            incidence[right].append((left, edge))
        neighbors, edge_ids = np.asarray([entry for row in incidence for entry in row]).T
        self.offsets = self.integers(np.cumsum([0] + [len(row) for row in incidence]))
        self.connectivity = self.integers(neighbors)
        self.material = self.rest.new_tensor(np.stack((cloth.lengths[edge_ids],
                                                       cloth.stiffness[edge_ids] * stiffness_scale), axis=1))
        self.config = self.integers([len(cloth.rest), substeps])


class MetalXPBD(MetalSolver):
    def __init__(self, cloth, batch, substeps, iterations, device):
        super().__init__(cloth, batch, substeps, device, 0.4)
        self.kernel = self.library.xpbd_step
        self.iterations = iterations
        groups = edge_color_groups(cloth)
        if len(groups) > 32:
            raise ValueError("fused Metal XPBD supports up to 32 constraint colors")
        order = np.concatenate(groups)
        self.offsets = self.integers(np.cumsum([0] + [len(group) for group in groups]))
        self.connectivity = self.integers(cloth.edges[order])
        self.material = self.rest.new_tensor(np.stack((cloth.lengths[order], 1 / cloth.stiffness[order]), axis=1))
        self.config = self.integers([len(cloth.rest), substeps, iterations, len(groups)])

"""Batched float32 physics on PyTorch CPU, CUDA or Apple Metal (MPS)."""

from dataclasses import dataclass
import math
from time import perf_counter_ns

import numpy as np
import torch

from cloth_sim import edge_color_groups


def select_device(name):
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if name == "auto":
        name = next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available[name]:
        raise ValueError(f"PyTorch does not see a usable {name} device")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class Grasp:
    indices: torch.Tensor
    free: torch.Tensor
    targets: torch.Tensor
    slots: torch.Tensor


def prepare_actions(cloth, corners, targets, held, device):
    """Transfer the entire action schedule before running the simulation."""
    indices = torch.as_tensor(corners, dtype=torch.long, device=device)
    all_targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    masks = {}
    slots = {}
    for count in (2, 4):
        vertex_slots = np.full(len(cloth.rest), -1, dtype=np.int32)
        vertex_slots[corners[:count]] = np.arange(count)
        slots[count] = torch.as_tensor(vertex_slots, device=device)
        masks[count] = torch.as_tensor(vertex_slots < 0, device=device)
    actions = []
    for frame, grasped in enumerate(held):
        count = 4 if grasped else 2
        frame_targets = all_targets[frame, :, :count].contiguous()
        actions.append(Grasp(indices[:count], masks[count], frame_targets, slots[count]))
    return actions


def table_contact(previous, proposed, free, friction):
    penetration = (-proposed[..., 2]).clamp_min(0) * free
    slip = proposed[..., :2] - previous[..., :2]
    distance = torch.linalg.vector_norm(slip, dim=-1).clamp_min(1e-12)
    removal = (friction * penetration / distance).clamp_max(1)
    xy = proposed[..., :2] - slip * removal.unsqueeze(-1)
    return torch.cat((xy, (proposed[..., 2] + penetration).unsqueeze(-1)), dim=-1)


class TorchSolver:
    def __init__(self, cloth, batch, substeps, device, friction=0.4):
        self.device = device
        self.substeps = substeps
        self.friction = friction
        self.inverse_mass = cloth.inverse_mass
        self.rest = torch.as_tensor(cloth.rest, dtype=torch.float32, device=device)
        self.batch = batch
        self.gravity = self.rest.new_tensor([0, 0, -9.81])
        self.reset()

    def reset(self):
        self.positions = self.rest.unsqueeze(0).expand(self.batch, -1, -1)
        self.velocities = torch.zeros_like(self.positions)

    def step(self, dt, action):
        h = dt / self.substeps
        damping = math.exp(-2.0 * h)
        positions, velocities = self.positions, self.velocities
        start = positions.index_select(1, action.indices)
        grasp_delta = action.targets - start
        inverse_mass = action.free * self.inverse_mass
        solve_data = self.prepare_step(inverse_mass, h)
        for substep in range(1, self.substeps + 1):
            target = start + grasp_delta * (substep / self.substeps)
            velocity = velocities * damping + self.gravity * h
            predicted = self.substep(positions, velocity, h, action.indices, target, solve_data)
            constrained = table_contact(positions, predicted, action.free, self.friction)
            velocities = (constrained - positions) / h
            positions = constrained
        self.positions, self.velocities = positions, velocities
        return positions

    def prepare_step(self, inverse_mass, h):
        return inverse_mass


class TorchMassSpring(TorchSolver):
    def __init__(self, cloth, batch, substeps, device, stiffness_scale=1.0, friction=0.4):
        super().__init__(cloth, batch, substeps, device, friction)
        self.left, self.right = torch.as_tensor(cloth.edges.T.copy(), device=device)
        self.lengths = self.rest.new_tensor(cloth.lengths)
        self.stiffness = self.rest.new_tensor(cloth.stiffness * stiffness_scale)

    def substep(self, positions, velocity, h, indices, target, inverse_mass):
        displacement = positions.index_select(1, self.right) - positions.index_select(1, self.left)
        length = torch.linalg.vector_norm(displacement, dim=-1)
        scale = self.stiffness * (length - self.lengths) / length.clamp_min(1e-12)
        force = displacement * scale.unsqueeze(-1)
        total = torch.zeros_like(positions)
        total.index_add_(1, self.left, force)
        total.index_add_(1, self.right, -force)
        velocity = velocity + total * inverse_mass[:, None] * h
        return (positions + velocity * h).index_copy_(1, indices, target)


class TorchXPBD(TorchSolver):
    def __init__(self, cloth, batch, substeps, iterations, device):
        super().__init__(cloth, batch, substeps, device)
        self.iterations = iterations
        self.groups = []
        for group in edge_color_groups(cloth):
            left, right = torch.as_tensor(cloth.edges[group].T.copy(), device=device)
            self.groups.append((left, right, self.rest.new_tensor(cloth.lengths[group]),
                                self.rest.new_tensor(1 / cloth.stiffness[group])))

    def prepare_step(self, inverse_mass, h):
        groups = []
        for left, right, rest, compliance in self.groups:
            w_left = inverse_mass.index_select(0, left)
            w_right = inverse_mass.index_select(0, right)
            alpha = compliance / h**2
            groups.append((left, right, rest, w_left, w_right, alpha, 1 / (w_left + w_right + alpha)))
        return groups

    def substep(self, positions, velocity, h, indices, target, groups):
        proposed = (positions + velocity * h).index_copy_(1, indices, target)
        # Multipliers are reset each substep and retained between sweeps.
        multipliers = [positions.new_zeros((len(positions), len(group[0]))) for group in groups]
        for _ in range(self.iterations):
            for index, (left, right, rest, w_left, w_right, alpha, reciprocal) in enumerate(groups):
                x_left, x_right = proposed.index_select(1, left), proposed.index_select(1, right)
                displacement = x_right - x_left
                length = torch.linalg.vector_norm(displacement, dim=-1)
                change = (rest - length - alpha * multipliers[index]) * reciprocal
                multipliers[index] += change
                correction = displacement * (change / length.clamp_min(1e-12)).unsqueeze(-1)
                proposed.index_copy_(1, left, x_left - w_left[:, None] * correction)
                proposed.index_copy_(1, right, x_right + w_right[:, None] * correction)
        return proposed


@torch.no_grad()
def rollout(solver, dt, actions, save_velocities=False):
    solver.reset()
    positions = [solver.positions]
    velocities = [solver.velocities]
    for action in actions:
        positions.append(solver.step(dt, action))
        if save_velocities:
            velocities.append(solver.velocities)
    return torch.stack(positions), torch.stack(velocities) if save_velocities else None


@torch.no_grad()
def benchmark(solver, dt, actions, warmup=8, throughput_block=16):
    device = solver.device
    # Warm both grasp phases so GPU graph setup stays out of the measurement.
    for action in actions[:warmup] + actions[-warmup:]:
        solver.step(dt, action)
    solver.reset()
    synchronize(device)

    latencies = []
    for action in actions:
        start = perf_counter_ns()
        solver.step(dt, action)
        synchronize(device)
        latencies.append((perf_counter_ns() - start) / 1e6)

    solver.reset()
    synchronize(device)
    start = perf_counter_ns()
    for start_frame in range(0, len(actions), throughput_block):
        for action in actions[start_frame:start_frame + throughput_block]:
            solver.step(dt, action)
        synchronize(device)
    throughput_seconds = (perf_counter_ns() - start) / 1e9

    # The quality pass is separate so no transfers or trajectory copies enter the timings.
    positions, velocities = rollout(solver, dt, actions, save_velocities=True)
    valid_frames = (torch.isfinite(positions) & torch.isfinite(velocities)).flatten(1).all(dim=1)
    invalid = np.flatnonzero(~valid_frames.cpu().numpy())
    if len(invalid):
        return None, {"failed_at_step": int(invalid[0])}
    return positions.cpu().numpy(), {
        "median_ms_per_batch_step": float(np.median(latencies)),
        "p95_ms_per_batch_step": float(np.percentile(latencies, 95)),
        "env_steps_per_sec": len(actions) * solver.batch / throughput_seconds,
        "throughput_sync_every_steps": throughput_block,
    }

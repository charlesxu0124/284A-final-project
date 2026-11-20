"""Batched cloth solvers with a shared mesh, table and kinematic grasps."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cloth:
    rest: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    lengths: np.ndarray
    stiffness: np.ndarray
    stretch_edge_count: int
    inverse_mass: float

    @classmethod
    def square(cls, resolution, side=0.4, density=0.20):
        axis = np.linspace(0, side, resolution)
        x, y = np.meshgrid(axis, axis)
        rest = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        vertices = np.arange(x.size).reshape(x.shape)
        a = vertices[:-1, :-1].ravel()
        b = vertices[:-1, 1:].ravel()
        c = vertices[1:, :-1].ravel()
        d = vertices[1:, 1:].ravel()
        triangles = np.stack((np.stack((a, b, d), axis=1),
                              np.stack((a, d, c), axis=1)), axis=1).reshape(-1, 3)

        def pairs(left, right):
            return np.column_stack((left.ravel(), right.ravel()))

        stretch = np.concatenate((pairs(vertices[:, :-1], vertices[:, 1:]),
                                  pairs(vertices[:-1, :], vertices[1:, :])))
        shear = np.concatenate((pairs(a, d), pairs(b, c)))
        bend = np.concatenate((pairs(vertices[:, :-2], vertices[:, 2:]),
                               pairs(vertices[:-2, :], vertices[2:, :])))
        edges = np.concatenate((stretch, shear, bend))
        lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
        stiffness = np.repeat([35.0, 22.0, 2.5], [len(stretch), len(shear), len(bend)])
        return cls(rest, triangles, edges, lengths, stiffness, len(stretch),
                   x.size / (density * side**2))


class ClothSolver:
    def __init__(self, cloth, batch, substeps):
        self.cloth = cloth
        self.substeps = substeps
        self.positions = np.repeat(cloth.rest[None, :, :], batch, axis=0)
        self.velocities = np.zeros_like(self.positions)
        self.left, self.right = cloth.edges.T

    def step(self, dt, held, targets):
        """Advance one control step. Targets are ordered like the held vertex IDs."""
        h = dt / self.substeps
        start = self.positions[:, held].copy()
        inverse_mass = np.full(len(self.cloth.rest), self.cloth.inverse_mass)
        inverse_mass[held] = 0.0
        damping = np.exp(-2.0 * h)
        for substep in range(1, self.substeps + 1):
            previous = self.positions
            velocity = self.velocities * damping
            velocity[:, :, 2] -= 9.81 * h
            proposed = self.predict(previous, velocity, inverse_mass, h)
            proposed[:, held] = start + (targets - start) * (substep / self.substeps)
            self.project(proposed, inverse_mass, h)
            self.apply_table(previous, proposed, inverse_mass)
            self.positions = proposed
            self.velocities = (proposed - previous) / h
        return self.positions

    @staticmethod
    def apply_table(previous, proposed, inverse_mass):
        # Inelastic plane contact with a tangential correction proportional to penetration.
        penetration = np.maximum(-proposed[:, :, 2], 0.0) * (inverse_mass > 0)
        slip = proposed[:, :, :2] - previous[:, :, :2]
        distance = np.linalg.norm(slip, axis=2)
        correction = np.minimum(0.40 * penetration / np.maximum(distance, 1e-12), 1.0)
        proposed[:, :, :2] -= slip * correction[:, :, None]
        proposed[:, :, 2] += penetration


class MassSpring(ClothSolver):
    def predict(self, positions, velocity, inverse_mass, h):
        displacement = positions[:, self.right] - positions[:, self.left]
        length = np.linalg.norm(displacement, axis=2)
        extension = length - self.cloth.lengths
        force = displacement * (self.cloth.stiffness * extension /
                                np.maximum(length, 1e-12))[:, :, None]
        total = np.zeros_like(positions)
        np.add.at(total, (slice(None), self.left), force)
        np.add.at(total, (slice(None), self.right), -force)
        velocity += total * inverse_mass[None, :, None] * h
        return positions + velocity * h

    def project(self, positions, inverse_mass, h):
        pass


class XPBD(ClothSolver):
    def __init__(self, cloth, batch, substeps, iterations):
        super().__init__(cloth, batch, substeps)
        self.iterations = iterations
        self.groups = [(self.left[group], self.right[group], cloth.lengths[group],
                        1.0 / cloth.stiffness[group]) for group in edge_color_groups(cloth)]

    def predict(self, positions, velocity, inverse_mass, h):
        return positions + velocity * h

    def project(self, positions, inverse_mass, h):
        constraints = []
        for left, right, rest, compliance in self.groups:
            alpha = compliance / h**2
            w_left, w_right = inverse_mass[left], inverse_mass[right]
            denominator = w_left + w_right + alpha
            multipliers = np.zeros((len(positions), len(left)))
            constraints.append((left, right, rest, w_left, w_right, alpha,
                                denominator, multipliers))

        for _ in range(self.iterations):
            for left, right, rest, w_left, w_right, alpha, denominator, multipliers in constraints:
                displacement = positions[:, right] - positions[:, left]
                length = np.linalg.norm(displacement, axis=2)
                change = (rest - length - alpha * multipliers) / denominator
                multipliers += change
                correction = displacement * (change / np.maximum(length, 1e-12))[:, :, None]
                positions[:, left] -= w_left[None, :, None] * correction
                positions[:, right] += w_right[None, :, None] * correction


def edge_color_groups(cloth):
    """Group constraints with disjoint endpoints, preserving order across backends."""
    used_colors = [set() for _ in cloth.rest]
    groups = []
    for edge, (left, right) in enumerate(cloth.edges):
        color = 0
        while color in used_colors[left] or color in used_colors[right]:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(edge)
        used_colors[left].add(color)
        used_colors[right].add(color)
    return groups

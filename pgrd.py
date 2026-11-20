"""PGRD adapted to a fixed cloth mesh and known grasps."""

import math

import numpy as np
import torch
from torch import nn

from torch_sim import TorchMassSpring, table_contact


def mesh_neighborhoods(cloth):
    adjacent = [{index} for index in range(len(cloth.rest))]
    for triangle in cloth.triangles:
        for index in triangle:
            adjacent[index].update(triangle)
    width = max(map(len, adjacent))
    neighbors = np.empty((len(adjacent), width), dtype=np.int64)
    valid = np.zeros_like(neighbors, dtype=bool)
    for index, row in enumerate(adjacent):
        neighbors[index] = index
        neighbors[index, :len(row)] = sorted(row)
        valid[index, :len(row)] = True
    return torch.from_numpy(neighbors), torch.from_numpy(valid)


class TemporalBlock(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, tokens):
        count, window, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens))
        q, k, v = qkv.reshape(count, window, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(count, window, width)
        tokens = tokens + self.output(context)
        return tokens + self.mlp(self.mlp_norm(tokens))


class VelocityResidual(nn.Module):
    def __init__(self, cloth, width=32, window=4):
        super().__init__()
        self.window = window
        neighbors, valid = mesh_neighborhoods(cloth)
        self.register_buffer("neighbors", neighbors, persistent=False)
        self.register_buffer("valid_neighbors", valid, persistent=False)
        self.register_buffer("frequencies", math.pi * 2.0**torch.arange(4), persistent=False)
        self.side = float(cloth.rest[:, 0].max())

        self.encode = nn.Linear(18, width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.relative_bias = nn.Linear(3, 1, bias=False)
        self.spatial_norm = nn.LayerNorm(width)
        self.decode = nn.Sequential(nn.Linear(width + 24, width), nn.GELU(), nn.Linear(width, 3))

        self.time_embed = nn.Linear(3, width)
        frequencies = 10000.0 ** (-2.0 * torch.arange(width // 2) / width)
        angles = torch.arange(window)[:, None] * frequencies
        time_code = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        self.register_buffer("time_code", time_code, persistent=False)
        self.temporal = nn.Sequential(TemporalBlock(width), TemporalBlock(width))
        self.delta = nn.Linear(width, 3)
        self.gate = nn.Linear(width, 3)
        # Start at the calibrated physics prediction.
        for layer in (self.decode[-1], self.delta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, current, previous, simulated, history):
        x, v = current
        x_prev, v_prev = previous
        x_sim, v_sim = simulated
        state = torch.cat((x / self.side, v, x_prev / self.side, v_prev, x_sim / self.side, v_sim), dim=-1)
        features = self.encode(state)
        batch, count, width = features.shape
        flat_neighbors = self.neighbors.flatten()
        neighbor_features = self.key_value(features).index_select(1, flat_neighbors)
        keys, values = neighbor_features.reshape(batch, count, -1, 2 * width).chunk(2, dim=-1)
        neighbors_xyz = x.index_select(1, flat_neighbors).reshape(batch, count, -1, 3)
        relative = (neighbors_xyz - x.unsqueeze(2)) / self.side
        logits = (self.query(features).unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(width)
        logits = logits + self.relative_bias(relative).squeeze(-1)
        logits = logits.masked_fill(~self.valid_neighbors, -torch.inf)
        encoded = self.spatial_norm(features + (logits.softmax(dim=2).unsqueeze(-1) * values).sum(dim=2))
        angles = (x / self.side).unsqueeze(-1) * self.frequencies
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        proposal = self.decode(torch.cat((encoded, fourier), dim=-1))

        history = (*history, proposal)[-self.window:]
        padded = (proposal,) * (self.window - len(history)) + history
        sequence = torch.stack(padded, dim=2).reshape(batch * count, self.window, 3)
        context = self.temporal(self.time_embed(sequence) + self.time_code)[:, -1]
        gate = self.gate(context).sigmoid().reshape(batch, count, 3)
        delta = self.delta(context).tanh().reshape(batch, count, 3)
        return 0.1 * ((1 - gate) * proposal + gate * delta), history


class PGRD:
    def __init__(self, backbone, model):
        self.backbone = backbone
        self.model = model
        self.device = backbone.device
        self.batch = backbone.batch
        self.substeps = backbone.substeps
        self.reset()

    def reset(self):
        self.backbone.reset()
        self.set_state(self.backbone.positions, self.backbone.velocities)

    def set_state(self, positions, velocities, previous=None):
        self.positions = positions
        self.velocities = velocities
        self.previous = (positions, velocities) if previous is None else previous
        self.history = ()

    def step(self, dt, action):
        current = self.positions, self.velocities
        # The physics solver is frozen. Predictions still depend on the corrected rollout state.
        with torch.no_grad():
            self.backbone.positions, self.backbone.velocities = current
            x_sim = self.backbone.step(dt, action)
            v_sim = self.backbone.velocities
        dv, self.history = self.model(current, self.previous, (x_sim, v_sim), self.history)
        dv = dv * action.free[:, None]
        proposed = x_sim + dt * dv
        # Only the residual displacement is projected here. The backbone already handles contact.
        x_next = table_contact(x_sim, proposed, action.free, self.backbone.friction)
        self.positions = x_next
        self.velocities = v_sim + (x_next - x_sim) / dt
        self.previous = current
        return self.positions


def from_checkpoint(path, cloth, batch, device, substeps=None, spring_class=TorchMassSpring):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("version") != 1:
        raise ValueError("expected a version 1 cloth PGRD checkpoint")
    settings = data["physics"]
    substeps = settings["substeps"] if substeps is None else substeps
    backbone = spring_class(cloth, batch, substeps, device,
                            stiffness_scale=settings["stiffness_scale"], friction=settings["friction"])
    model = VelocityResidual(cloth, **data["model_config"]).to(device)
    model.load_state_dict(data["model"])
    return PGRD(backbone, model.eval()), data

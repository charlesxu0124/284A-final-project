"""Calibrate spring physics and train PGRD on synthetic tabletop folds."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from benchmark import DT, fold_actions, positive_int, quality, scene_fields
from cloth_sim import Cloth
from pgrd import PGRD, VelocityResidual
from torch_sim import TorchMassSpring, TorchXPBD, prepare_actions, rollout, select_device


@dataclass(frozen=True)
class Split:
    corners: np.ndarray
    targets: np.ndarray
    held: np.ndarray
    cpu_actions: list
    positions: torch.Tensor
    velocities: torch.Tensor | None

    def actions_on(self, cloth, device):
        return prepare_actions(cloth, self.corners, self.targets, self.held, device)


def make_split(cloth, args, size, name):
    device = torch.device("cpu")
    corners, targets, held = fold_actions(cloth, args.resolution, args.steps, size, name, args.seed)
    actions = prepare_actions(cloth, corners, targets, held, device)
    teacher = TorchXPBD(cloth, size, args.teacher_substeps, args.teacher_iterations, device)
    positions, velocities = rollout(teacher, DT, actions, save_velocities=name == "train")
    if not torch.isfinite(positions).all() or (velocities is not None and not torch.isfinite(velocities).all()):
        raise RuntimeError(f"XPBD teacher diverged on the {name} set")
    return Split(corners, targets, held, actions, positions, velocities)


def free_vertex_loss(predicted, reference, free):
    # free is [frames, vertices]. The trajectory arrays include one batch dimension.
    squared_mm = ((1000.0 * (predicted - reference))**2).sum(dim=-1)
    return (squared_mm * free[:, None]).sum() / (predicted.shape[1] * free.sum())


def calibrate(cloth, training, substeps):
    free = torch.stack([action.free for action in training.cpu_actions])
    trials = []
    for scale, friction in product((0.8, 1.0, 1.25), (0.25, 0.40, 0.55)):
        parameters = {"stiffness_scale": scale, "friction": friction, "substeps": substeps}
        solver = TorchMassSpring(cloth, training.targets.shape[1], device=torch.device("cpu"), **parameters)
        positions, _ = rollout(solver, DT, training.cpu_actions)
        rmse = float(free_vertex_loss(positions[1:], training.positions[1:], free).sqrt())
        trials.append({"physics": parameters, "train_rmse_mm": rmse if math.isfinite(rmse) else None})
    valid = [trial for trial in trials if trial["train_rmse_mm"] is not None]
    if not valid:
        raise RuntimeError("all spring fits diverged. Increase --spring-substeps")
    return min(valid, key=lambda trial: trial["train_rmse_mm"])["physics"], trials


def train(model, cloth, physics, training, validation, args, device):
    train_solver = PGRD(TorchMassSpring(cloth, args.train_folds, device=device, **physics), model)
    val_solver = PGRD(TorchMassSpring(cloth, args.validation_folds, device=device, **physics), model)
    train_actions = training.actions_on(cloth, device)
    val_actions = validation.actions_on(cloth, device)
    train_x = training.positions.to(device)
    train_v = training.velocities.to(device)
    val_x = validation.positions.to(device)
    train_free = torch.stack([action.free for action in train_actions])
    val_free = torch.stack([action.free for action in val_actions])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 3]))

    def validate(update):
        model.eval()
        positions, _ = rollout(val_solver, DT, val_actions)
        error = float(free_vertex_loss(positions[1:], val_x[1:], val_free).sqrt())
        if not math.isfinite(error):
            raise RuntimeError(f"nonfinite validation error at update {update}")
        print(f"Update {update:4d}   validation rollout RMSE {error:.3f} mm", flush=True)
        return error

    best_error = validate(0)
    best_update = 0
    best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    history = [{"update": 0, "validation_rmse_mm": best_error}]
    for update in range(1, args.updates + 1):
        model.train()
        start = int(rng.integers(0, args.steps - args.horizon + 1))
        end = start + args.horizon
        previous = max(start - 1, 0)
        train_solver.set_state(train_x[start], train_v[start], (train_x[previous], train_v[previous]))
        predicted = torch.stack([train_solver.step(DT, action) for action in train_actions[start:end]])
        loss = free_vertex_loss(predicted, train_x[start + 1:end + 1], train_free[start:end])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if update % args.validate_every == 0 or update == args.updates:
            error = validate(update)
            history.append({"update": update, "validation_rmse_mm": error})
            if error < best_error:
                best_error = error
                best_update = update
                best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return best_weights, best_update, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--resolution", type=positive_int, default=8)
    parser.add_argument("--steps", type=positive_int, default=120)
    parser.add_argument("--train-folds", type=positive_int, default=12)
    parser.add_argument("--validation-folds", type=positive_int, default=4)
    parser.add_argument("--test-folds", type=positive_int, default=8)
    parser.add_argument("--teacher-substeps", type=positive_int, default=4)
    parser.add_argument("--teacher-iterations", type=positive_int, default=12)
    parser.add_argument("--spring-substeps", type=positive_int, default=12)
    parser.add_argument("--updates", type=positive_int, default=200)
    parser.add_argument("--validate-every", type=positive_int, default=25)
    parser.add_argument("--horizon", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/pgrd-training"))
    args = parser.parse_args()
    if args.resolution < 3 or args.steps < 8 or args.seed < 0 or args.horizon > args.steps:
        parser.error("use resolution >= 3, steps >= 8, seed >= 0 and horizon <= steps")
    try:
        device = select_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cloth = Cloth.square(args.resolution)
    args.output.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    training = make_split(cloth, args, args.train_folds, "train")
    validation = make_split(cloth, args, args.validation_folds, "validation")
    test = make_split(cloth, args, args.test_folds, "test")
    teacher_seconds = perf_counter() - start
    start = perf_counter()
    physics, calibration = calibrate(cloth, training, args.spring_substeps)
    calibration_seconds = perf_counter() - start
    print(f"Calibrated spring model {physics}. Training residuals on {device}.", flush=True)

    model_config = {"width": 32, "window": 4}
    model = VelocityResidual(cloth, **model_config).to(device)
    start = perf_counter()
    weights, best_update, history = train(model, cloth, physics, training, validation, args, device)
    residual_seconds = perf_counter() - start
    config = {key: value for key, value in vars(args).items() if key != "output"}
    torch.save({"version": 1, "model": weights, "model_config": model_config, "physics": physics,
                "training_config": config, "best_update": best_update}, args.output / "pgrd.pt")

    actions = test.actions_on(cloth, device)
    spring = TorchMassSpring(cloth, args.test_folds, device=device, **physics)
    teacher = test.positions.numpy()
    test_metrics = {}
    for name, solver in (("spring", spring), ("pgrd", PGRD(spring, model.eval()))):
        positions, _ = rollout(solver, DT, actions)
        test_metrics[name] = quality(positions.cpu().numpy(), cloth, args.resolution,
                                     test.corners, test.held, teacher)

    np.savez_compressed(args.output / "test-reference.npz", positions=teacher,
                        **scene_fields(cloth, test.corners, test.targets, test.held))
    report = {"device": str(device), "torch_version": str(torch.__version__), "config": config,
              "teacher_seconds": teacher_seconds, "calibration_seconds": calibration_seconds,
              "residual_seconds": residual_seconds, "physics": physics, "calibration": calibration,
              "best_update": best_update, "validation": history, "test": test_metrics}
    (args.output / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

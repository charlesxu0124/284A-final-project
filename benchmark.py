"""Benchmark a scripted tabletop fold with prescribed cloth corner grasps."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from cloth_sim import Cloth, MassSpring, XPBD

DT = 1.0 / 60.0


def fold_actions(cloth, resolution, steps, batch, split="scripted", seed=7):
    corners = np.array([len(cloth.rest) - resolution, len(cloth.rest) - 1, 0, resolution - 1])
    progress = np.arange(1, steps + 1) / steps
    if split == "scripted":
        end = np.full(batch, 0.75)
        height = 0.20 + 0.03 * np.sin(np.arange(batch))
    else:
        split_id = ("train", "validation", "test").index(split)
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_id]))
        # Use one draw per environment so larger batches preserve smaller batches as a prefix.
        height, end = rng.uniform([0.14, 0.63], [0.26, 0.79], size=(batch, 2)).T
    phase = np.clip(progress[:, None] / end, 0.0, 1.0)
    phase = phase**2 * (3.0 - 2.0 * phase)
    targets = np.tile(cloth.rest[corners], (steps, batch, 1, 1))
    targets[:, :, 2:, 1] = (cloth.rest[-1, 1] * phase)[:, :, None]
    lift = 4.0 * phase * (1.0 - phase) * height
    targets[:, :, 2:, 2] = lift[:, :, None]
    held = progress <= 0.85
    return corners, targets, held


def scene_fields(cloth, corners, targets, held):
    return {"rest": cloth.rest, "triangles": cloth.triangles, "dt": DT,
            "corners": corners, "targets": targets, "moving_corners_held": held}


@np.errstate(over="ignore", invalid="ignore")
def run(solver, dt, corners, targets, held, warmup):
    initial_positions = solver.positions.copy()
    initial_velocities = solver.velocities.copy()
    for grasped, target in zip(held[:warmup], targets[:warmup]):
        count = 4 if grasped else 2
        solver.step(dt, corners[:count], target[:, :count])
    solver.positions, solver.velocities = initial_positions, initial_velocities

    trajectory = np.empty((len(targets) + 1, *initial_positions.shape), dtype=initial_positions.dtype)
    trajectory[0] = initial_positions
    timings = []
    for frame, (grasped, target) in enumerate(zip(held, targets), start=1):
        count = 4 if grasped else 2
        indices, target = corners[:count], target[:, :count]
        start = perf_counter_ns()
        positions = solver.step(dt, indices, target)
        elapsed = (perf_counter_ns() - start) / 1e6
        if not np.isfinite(positions).all() or not np.isfinite(solver.velocities).all():
            return None, {"failed_at_step": frame}
        trajectory[frame] = positions
        timings.append(elapsed)

    return trajectory, {
        "median_ms_per_batch_step": float(np.median(timings)),
        "p95_ms_per_batch_step": float(np.percentile(timings, 95)),
        "env_steps_per_sec": (1000.0 * len(timings) * len(initial_positions) / sum(timings)),
    }


def quality(positions, cloth, resolution, corners, held, reference):
    vertices = np.arange(len(cloth.rest)).reshape(resolution, resolution)
    lower = vertices[:resolution // 2, 1:-1].ravel()
    upper = vertices[::-1][:resolution // 2, 1:-1].ravel()
    # Mid-plane alignment in the table's XY coordinates, independent of layer thickness.
    final_xy = positions[-1, :, :, :2]
    offset = final_xy[:, lower] - final_xy[:, upper]
    edges = cloth.edges[:cloth.stretch_edge_count]
    lengths = np.linalg.norm(positions[1:, :, edges[:, 1]] - positions[1:, :, edges[:, 0]], axis=3)
    stretch = np.maximum(lengths / cloth.lengths[:cloth.stretch_edge_count] - 1.0, 0.0)
    result = {
        "final_fold_alignment_mm": float(np.linalg.norm(offset, axis=2).mean() * 1000.0),
        "p95_edge_stretch_pct": float(np.percentile(stretch, 95) * 100.0),
    }
    if reference is not None:
        squared_distance = np.sum((positions[1:] - reference[1:])**2, axis=3)
        free = np.ones((len(held), len(cloth.rest)), dtype=bool)
        free[:, corners[:2]] = False
        free[np.ix_(held, corners[2:])] = False
        result["rollout_rmse_mm"] = float(np.sqrt((squared_distance * free[:, None, :]).sum() /
                                                  (free.sum() * positions.shape[1])) * 1000.0)
        result["final_rmse_mm"] = float(np.sqrt(squared_distance[-1][:, free[-1]].mean()) * 1000.0)
    return result


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("both", "spring", "xpbd", "pgrd", "all"), default="both")
    parser.add_argument("--device", choices=("numpy", "cpu", "mps", "cuda", "auto"),
                        help="defaults to NumPy, or auto when a checkpoint is supplied")
    parser.add_argument("--physics-engine", choices=("torch", "metal"), default="torch",
                        help="metal fuses the physics substeps on Apple GPUs")
    parser.add_argument("--checkpoint", type=Path, help="PGRD checkpoint, also calibrates the spring control")
    parser.add_argument("--resolution", type=positive_int, default=12, help="vertices per side")
    parser.add_argument("--batch", type=positive_int, default=1, help="independent cloth environments")
    parser.add_argument("--steps", type=positive_int, default=120)
    parser.add_argument("--spring-substeps", type=positive_int, help="defaults to 12 or the checkpoint setting")
    parser.add_argument("--xpbd-substeps", type=positive_int, default=2)
    parser.add_argument("--iterations", type=positive_int, default=4, help="XPBD sweeps per substep")
    parser.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU thread count")
    parser.add_argument("--split", choices=("scripted", "train", "validation", "test"), default="scripted")
    parser.add_argument("--seed", type=int, default=7, help="seed for the randomized folds")
    parser.add_argument("--reference", type=Path, help="aligned per-vertex reference NPZ")
    parser.add_argument("--output", type=Path, help="directory for trajectories and metrics")
    args = parser.parse_args()
    if args.resolution < 3:
        parser.error("resolution must be at least 3")
    if args.steps < 8:
        parser.error("steps must be at least 8 to include lifting, placing and release")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.solver in ("pgrd", "all") and not args.checkpoint:
        parser.error("PGRD needs a checkpoint from train_pgrd.py")
    device = args.device
    if device is None:
        if args.physics_engine == "metal":
            device = "mps"
        else:
            device = "auto" if args.checkpoint else "numpy"
    if device == "numpy" and args.checkpoint:
        parser.error("checkpoint evaluation requires --device cpu, mps, cuda or auto")
    if args.physics_engine == "metal" and (device not in ("mps", "auto") or args.resolution > 32):
        parser.error("fused Metal physics needs an MPS device and resolution <= 32")

    cloth = Cloth.square(args.resolution)
    corners, targets, held = fold_actions(cloth, args.resolution, args.steps, args.batch, args.split, args.seed)
    scene = scene_fields(cloth, corners, targets, held)
    reference = None
    if args.reference:
        with np.load(args.reference) as data:
            for key, value in scene.items():
                if key not in data or np.shape(data[key]) != np.shape(value):
                    parser.error(f"reference must match the scripted scene ({key})")
                if not np.allclose(data[key], value, rtol=0.0, atol=1e-7):
                    parser.error(f"reference must match the scripted scene ({key})")
            if "positions" not in data:
                parser.error("reference must contain positions")
            reference = data["positions"]
        shape = (args.steps + 1, args.batch, len(cloth.rest), 3)
        if reference.shape != shape or not np.isfinite(reference).all():
            parser.error(f"reference positions must be finite with shape {shape}")
        if not np.allclose(reference[0], cloth.rest, rtol=0.0, atol=1e-7):
            parser.error("reference must start from the scripted rest state")

    names = {"both": ("spring", "xpbd"), "all": ("spring", "xpbd", "pgrd")}.get(args.solver, (args.solver,))
    warmup = 8
    if device == "numpy":
        factories = {"spring": lambda: MassSpring(cloth, args.batch, args.spring_substeps or 12),
                     "xpbd": lambda: XPBD(cloth, args.batch, args.xpbd_substeps, args.iterations)}
        execute = lambda solver: run(solver, DT, corners, targets, held, warmup)
        device_info = {"device": "numpy", "dtype": "float64"}
    else:
        try:
            import torch
            from torch_sim import TorchMassSpring, TorchXPBD, benchmark, prepare_actions, select_device
            from pgrd import from_checkpoint
        except ImportError as exc:
            parser.error(f"install requirements-torch.txt for PyTorch devices ({exc})")
        try:
            torch_device = select_device(device)
        except ValueError as exc:
            parser.error(str(exc))
        spring_class, xpbd_class = TorchMassSpring, TorchXPBD
        if args.physics_engine == "metal":
            if torch_device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
                parser.error("fused Metal physics requires an MPS device and torch.mps.compile_shader")
            from metal_sim import MetalMassSpring, MetalXPBD
            spring_class, xpbd_class = MetalMassSpring, MetalXPBD
        torch.set_num_threads(args.threads)
        actions = prepare_actions(cloth, corners, targets, held, torch_device)
        physics = {"friction": 0.4, "stiffness_scale": 1.0, "substeps": 12}
        hybrid = None
        if args.checkpoint:
            hybrid, checkpoint = from_checkpoint(args.checkpoint, cloth, args.batch,
                                                torch_device, args.spring_substeps, spring_class)
            physics = checkpoint["physics"]
        substeps = args.spring_substeps or physics["substeps"]
        factories = {
            "spring": lambda: spring_class(cloth, args.batch, substeps, torch_device,
                                           physics["stiffness_scale"], physics["friction"]),
            "xpbd": lambda: xpbd_class(cloth, args.batch, args.xpbd_substeps, args.iterations, torch_device),
            "pgrd": lambda: hybrid,
        }
        execute = lambda solver: benchmark(solver, DT, actions, warmup)
        device_info = {"device": str(torch_device), "dtype": "float32", "torch_version": str(torch.__version__),
                       "torch_cpu_threads": args.threads, "physics_engine": args.physics_engine}
        if torch_device.type == "cuda":
            device_info["accelerator"] = torch.cuda.get_device_name(torch_device)
        elif torch_device.type == "mps" and hasattr(torch.backends.mps, "get_name"):
            device_info["accelerator"] = torch.backends.mps.get_name()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(), "numpy_version": np.__version__,
        "resolution": args.resolution, "batch": args.batch, "steps": args.steps,
        "dt": DT, "warmup_steps": warmup if device == "numpy" else 2 * warmup,
        "reference": str(args.reference) if args.reference else None,
        "split": args.split, "seed": args.seed, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        **device_info,
        "solvers": {},
    }
    failed = False
    for name in names:
        solver = factories[name]()
        trajectory, metrics = execute(solver)
        metrics["substeps"] = solver.substeps
        if name == "xpbd":
            metrics["iterations"] = solver.iterations
        elif device != "numpy":
            metrics.update({key: physics[key] for key in ("friction", "stiffness_scale")})
        if trajectory is None:
            failed = True
        else:
            metrics.update(quality(trajectory, cloth, args.resolution, corners, held, reference))
            if args.output:
                np.savez_compressed(args.output / f"{name}.npz", positions=trajectory, **scene)
        report["solvers"][name] = metrics

    summary = json.dumps(report, indent=2, allow_nan=False)
    print(summary)
    if args.output:
        (args.output / f"metrics-{args.solver}.json").write_text(summary + "\n", encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

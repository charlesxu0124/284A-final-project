# Cloth simulation handoff

## Where we are

We have three runnable methods on the same 40 cm tabletop fold, an example PGRD checkpoint, and a reproducible synthetic test set. The first experiment separates two questions. Does the residual improve a calibrated spring rollout, and how much throughput does each method get on a GPU?

| Method | What it does | Where to start |
| --- | --- | --- |
| Mass–spring | Semi-implicit Euler with stretch, shear and two-hop bending springs | `cloth_sim.py`, `TorchMassSpring` in `torch_sim.py` |
| XPBD | Compliant distance constraints with a sequential sweep over disjoint edge colors | `cloth_sim.py`, `TorchXPBD` in `torch_sim.py` |
| PGRD | Calibrated spring step plus a learned velocity correction for each free vertex | `pgrd.py`, `train_pgrd.py` |
| Fused Metal physics | One GPU threadgroup per cloth, with all the physics substeps inside one dispatch per control step | `metal_sim.py`, `cloth_kernels.metal` |
| Evaluation | Builds the fold, compares trajectories, and records latency and throughput | `benchmark.py`, `test_simulation.py` |

The scripts and install commands are in the [README](README.md).

## What we implemented from PGRD

The paper first fits mass–spring parameters to trajectories, then trains a residual that sees the current state, previous state and spring prediction. The residual predicts a velocity change, which is added to the simulated velocity and integrated into a position change. Held vertices receive zero correction. The paper uses a Fourier-conditioned decoder, a gated temporal transformer and multi-step rollout training. It reports lower tracking error on a flag held by a supporting stick. [PGRD paper and appendix](https://arxiv.org/html/2607.13451v1)

Our fixed-mesh adaptation uses one-ring attention over triangle neighbors in place of Point Transformer V3. It has a smaller decoder and a two-layer, four-frame temporal transformer. A nine-point grid search fits spring stiffness and table friction. During training the five-step windows feed predictions back into the solver, with gradients through the residual network and its history. The spring step is frozen. We project the residual displacement against the table after it is applied, then update velocity consistently with that projection. The authors' separate release provides the original flag configuration and checkpoints. The upstream stack and its flag evaluation still need a Linux/CUDA run. [Authors' implementation](https://github.com/shivanshpatel35/pgrd)

## Held-out rollout quality

The demo uses an 8 × 8 mesh and 120 control steps. The generated folds vary lift height and timing. Seed 7 generates disjoint sets of 12 training, 4 validation and 8 test trajectories. The CPU teacher is XPBD with four substeps and twelve sweeps per substep. Calibration sees training folds. Checkpoint selection uses full validation rollouts. The demo checkpoint was selected at update 125 of 200 on the Apple M5 Pro.

| Method | Rollout RMSE | Final-frame RMSE |
| --- | ---: | ---: |
| Calibrated mass–spring | 1.95 mm | 2.90 mm |
| PGRD | 1.76 mm | 3.61 mm |

The residual improves error over the full rollout by about 10% and increases terminal error by about 24%. The terminal result matters for a policy whose reward depends on the settled fold. The test data measures agreement with the XPBD teacher. This scene uses prescribed vertex grasps and a simple table plane. It omits cloth thickness and self-contact, so apparent layer alignment can occur even when the layers intersect. [Training report](artifacts/training.json)

We also trained with eight spring substeps using the same splits and validation rule. The hybrid reached 1.76 mm rollout and 3.37 mm final-frame RMSE, compared with 1.96 mm and 3.01 mm for its eight-substep spring control. With fused Metal at batch 64, the eight- and twelve-substep hybrids both ran at about 29k environment steps/s. Cutting four spring substeps made little difference once the spring computation was fused. To repeat the comparison, use `artifacts/pgrd-8substeps.pt` as the checkpoint with the same reference file. [Eight-substep training report](artifacts/training-8substeps.json)

## GPU results on Apple M5 Pro

Each environment has 64 vertices. Each control step advances the scene by 1/60 s. Mass–spring and the hybrid use 12 substeps. XPBD uses two substeps and four sweeps per substep. The calibrations, precision, folds and checkpoint are shared between the timing runs. CPU below means one PyTorch CPU thread. The GPU has two implementations. Portable PyTorch dispatches tensor operations individually. Fused Metal keeps each cloth's working positions in threadgroup memory and runs its physics in one kernel. PGRD's residual network still runs through PyTorch.

| Solver | Single-cloth median on CPU | Single-cloth median on fused Metal | Batch 64 on CPU | Batch 64 on PyTorch MPS | Batch 64 on fused Metal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mass–spring | 0.52 ms | 0.20 ms | 13.6k env steps/s | 33.9k env steps/s | 1.39M env steps/s |
| XPBD | 1.06 ms | 0.21 ms | 18.4k env steps/s | 11.2k env steps/s | 1.19M env steps/s |
| PGRD | 0.84 ms | 0.92 ms | 4.33k env steps/s | 15.9k env steps/s | 28.7k env steps/s |

The fused classical solvers are much faster on this small scene. At batch 64, fusion improves hybrid throughput about 1.8× over PyTorch MPS and 6.6× over one CPU thread. PGRD still takes far more compute than either fused physics baseline. Portable XPBD illustrates why GPU placement by itself can be slow. Each sequential edge color incurs a group of dispatches, so small batches can be dominated by launch overhead.

Latency synchronizes after every control step. Throughput synchronizes after blocks of 16 steps, keeping state and prescribed actions resident on the selected device. Warm-up, trajectory recording, CPU transfers for metrics and disk I/O are outside the timed region. The numbers cover cloth dynamics, prescribed grasps and table contact. They exclude policy inference, observation rendering and robot dynamics. The profile includes medians, p95s and batches of 1, 16, 64 and 256. It also includes a batch-256 CPU run with four PyTorch threads. These are single-machine measurements. [Raw profile](artifacts/profile-m5-pro.json)

## Reproducing and extending the experiment

- Use `--resolution 8 --batch 8 --split test --seed 7` with `artifacts/test-reference.npz`. The benchmark checks that the reference matches the mesh, action schedule and timestep. Rollout and final-frame RMSE exclude vertices held during the measured frame.
- Use `--split test` without `--reference` for batch scaling. The generated batch preserves the first eight held-out trajectories, then adds more trajectories from the same generator. `--output` saves a JSON report and an NPZ trajectory for each method. Each trajectory includes `[frame, environment, vertex, xyz]` in meters plus rest geometry and the action schedule.
- Run `python -m unittest -v test_simulation`. The suite checks that the CPU implementations agree, zero residual recovers mass–spring exactly, and train, validation and test trajectories are distinct. On an Apple GPU it also checks default PyTorch MPS and fused Metal against PyTorch CPU on the same fold.
- Keep mesh resolution fixed when comparing solvers. Explicit mass–spring may require more `--spring-substeps` at higher resolution. The bundled checkpoint was trained on an 8 × 8 grid.
- On NVIDIA, run the README's `--device cuda` command and compare the same checkpoint, dtype, resolution and batch sizes before optimizing. If portable XPBD is launch-bound, fuse the colored sweeps in Warp or CUDA and rerun the parity checks. The Apple kernels are in a separate file and use a threadgroup per cloth as a concrete design to start from.
- Use a contact-aware teacher such as C-IPC for the next synthetic results. Add self-intersection rate, error by rollout horizon and a settled-fold score. A real-world check can use tracked folds or the Blanco-Mulero table-contact data. [C-IPC](https://ipc-sim.github.io/C-IPC/) · [Cloth manipulation benchmark](https://arxiv.org/abs/2310.09543)
- For the final comparison, keep PGRD and its fitted physics control, add Newton VBD as a GPU physics baseline, and train a MeshGraphNets-style learned-only control on the same trajectories. To test whether learning buys speed, sweep solver budgets and compare throughput at matched rollout error and contact quality. [Newton VBD](https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_vbd.html) · [MeshGraphNets](https://arxiv.org/abs/2010.03409)

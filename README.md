# Learning Cloth

This project compares cloth simulation speed and rollout accuracy for robotic folding. A 40 cm square starts flat on a table. Two corners stay anchored while the opposite corners lift, cross the sheet, lower and release. Mass–spring, XPBD and a tabletop adaptation of Physics-Guided Residual Dynamics (PGRD) all use the same mesh, grasps and 60 Hz control interval.

## Start here

Use Python 3.11 or newer. The commands below use the pre-trained demo checkpoint and the eight held-out synthetic folds from the reported experiment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt
python -m unittest -v test_simulation
python benchmark.py --solver all --device cpu --resolution 8 --batch 8 --split test \
  --checkpoint artifacts/pgrd.pt --reference artifacts/test-reference.npz \
  --output results/cpu-demo
```

For a quick physics-only run on the original NumPy CPU backend, install `requirements.txt` and run `python benchmark.py`.

## Run on a GPU

**Apple Silicon.** Use the fused Metal kernels for physics. PGRD uses the fused spring step and runs its residual network in PyTorch on the same GPU. The environment variable keeps unsupported operations from silently falling back to the CPU.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmark.py --solver all --device mps \
  --physics-engine metal --resolution 8 --batch 64 --split test \
  --checkpoint artifacts/pgrd.pt --output results/metal-batch64
```

**NVIDIA.** Install the PyTorch build for your NVIDIA driver and use the portable PyTorch solver path. CUDA is wired up and still needs verification on an NVIDIA machine. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

```sh
python benchmark.py --solver all --device cuda --resolution 8 --batch 64 --split test \
  --checkpoint artifacts/pgrd.pt --output results/cuda-batch64
```

On either GPU, use `--solver both` and omit `--checkpoint` to run just the two classical methods with their stock parameters. With a checkpoint, `spring` uses the same calibrated parameters as PGRD. The fused Metal kernels support up to 32 vertices per side. Omit `--physics-engine metal` to benchmark the portable PyTorch implementation on Apple Silicon.

## Retrain PGRD

```sh
python train_pgrd.py --device auto --output results/retrained
```

This generates the synthetic folds, calibrates the spring model, trains the residual network and saves `pgrd.pt`, `test-reference.npz` and `training.json` in the output directory. Use `--help` on either script to see all options.

[Handoff notes](HANDOFF.md) cover the method, measured results, benchmark protocol, file map and next experiments.

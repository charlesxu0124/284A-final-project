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

## Render videos

`benchmark.py --output` saves each solver's trajectory. `render_video.py` draws them side by side (needs `matplotlib` and `ffmpeg`).

```sh
python benchmark.py --solver all --device cpu --resolution 8 --batch 8 --split test \
  --checkpoint artifacts/pgrd.pt --reference artifacts/test-reference.npz --output results/cpu-demo
python render_video.py results/cpu-demo/spring.npz results/cpu-demo/xpbd.npz results/cpu-demo/pgrd.npz \
  --labels "mass-spring" "XPBD" "PGRD" --output videos/solvers_8x8.mp4
```

`--environments 0 1 2 3` draws several folds of one file. Examples are in [videos/](videos/): the three solvers next to the XPBD teacher, XPBD at 8, 25 and 64 vertices per side, and PGRD on the eight held-out folds.

## MuJoCo cloth with robot arms and the 3D World Simulator

[mujoco_cloth/](mujoco_cloth/) simulates two robot arms manipulating a cloth in MuJoCo, renders every episode from four cameras, and fits a learned 3D world model to the result. The world model is the [3D World Simulator](https://github.com/jwang771/3d-world-simulator): an autoencoder turns the camera views into a voxel latent that decodes to Gaussians, and a dynamics model predicts the next latent from the last latents and the action. The cloth here is a MuJoCo flex object and has nothing in common with the solvers above.

### Setup

```sh
git clone https://github.com/jwang771/3d-world-simulator ~/3d-world-simulator    # follow its README to create the conda environment
conda activate 3dws
python -m pip install -r mujoco_cloth/requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie mujoco_cloth/mujoco_menagerie    # robot models
export WS_ROOT=~/3d-world-simulator
```

Rendering uses EGL (`MUJOCO_GL=egl`), so it runs on a headless machine with an NVIDIA GPU. Training needs a GPU with about 60 GB of memory at the default grid.

### Everything in one command

```sh
mujoco_cloth/run_3dws.sh fold      # or hold, grasp
```

This simulates the dataset, trains the autoencoder and the dynamics model, and writes open-loop rollout videos of held-out episodes to `results/mujoco/videos`. Expect about six hours for `fold` on one GPU. Settings such as `SIM_ARM`, `EPISODES` and `VAE_EPOCHS` are environment variables listed at the top of the script. The steps below are what it runs.

### 1. Generate data

```sh
SIM_ARM=yam SIM_SELFCOLLIDE=auto MUJOCO_GL=egl python mujoco_cloth/cloth_arms.py --mode fold \
  --out results/mujoco/data/simcloth_fold_yam --episodes 200 --frames 200 --val 20 --workers 16 --render-workers 5
```

| `--mode` | Episode |
| --- | --- |
| `hold` | Each arm holds one corner for the whole episode and the grippers move together |
| `grasp` | Each arm reaches for a point on the cloth, grasps it, carries it and lets go |
| `fold` | Several moves in a row at different speeds: two-arm and one-arm folds, lift and carry, drag |

`SIM_ARM` picks the arm, `yam` (I2RT YAM) or `vx300s` (ViperX 300 S, the default). `SIM_SELFCOLLIDE=auto` lets the cloth rest on itself, which folds need and which makes the simulation about three times slower. Episodes are 10 Hz. A grasp ties one cloth vertex to the fingertips while the gripper is closed, and the arms do not collide with the cloth.

The output folder holds `packed/episode_<E>.hdf5` (colour, depth, cloth mask, camera poses, actions and the simulator's joint angles), `train/` and `val/` splits, and `grid.json`, the workspace box and voxel grid fitted to where the cloth and grippers go. The action is, per arm, the commanded fingertip position and a closed flag.

### 2. Train the world model

Register the dataset as a task of the 3D World Simulator, then train from its folder.

```sh
python mujoco_cloth/register_task.py results/mujoco/data/simcloth_fold_yam
OUT=$PWD/results/mujoco; N=simcloth_fold_yam; R=$OUT/runs/$N
cd $WS_ROOT

python scripts/train_vae.py --recipe minimal --task $N --data-root $OUT/data --out-dir $R/vae \
  --n-epochs 30 --iters 300 --val-iters 20 --val-interval 5 --save-interval 5 --phase-a 8 --phase-b 8

for split in train val; do
  python scripts/export_latents.py --vae-ckpt $R/vae/checkpoints/latest.pth \
    --packed-dir $OUT/data/$N/$split --out-dir $R/latents/$split
done

python scripts/train_dynamics.py --latent-dir $R/latents/train --val-latent-dir $R/latents/val --out-dir $R/dynamics \
  --n-history 4 --ar-schedule 1 2 3 4 5 6 --ar-schedule-epochs 10 20 30 40 50 80 \
  --iters 300 --val-iters 20 --object-alpha 20 --save-interval 5
```

### 3. Open-loop rollouts

The world model gets the first frames of a held-out episode and then only the actions.

```sh
python scripts/export_release.py --out $R/task_198 --run $R/vae/checkpoints/latest.pth \
  $R/dynamics/checkpoints/latest.pth $R/latents/val/episode_198_latents.h5 $OUT/data/$N/val/episode_198.hdf5
python scripts/rollout.py --task $R/task_198 --frame 6 --steps 190 --view cams --compare --out $R/rollout_198.mp4
```

`--compare` puts the prediction above the true latent decoded by the autoencoder and prints the latent error per step.

### 4. Whole-scene videos

The world model only covers the workspace box and draws soft arms. These three steps rebuild the whole simulator image: a Gaussian model of the arms fitted to simulator renders, a static floor, and a small network that reads the joint angles from the predicted latent.

```sh
cd -    # back to this repository
python mujoco_cloth/fit_rig.py --states $OUT/data/$N/states --out $R/rig.pt
python mujoco_cloth/joint_head.py --latents $R/latents --packed $OUT/data/$N/packed --out $R/joint_head.pt
python mujoco_cloth/compose_video.py --task $R/task_198 --packed $OUT/data/$N/packed/episode_198.hdf5 \
  --rig $R/rig.pt --joint-head $R/joint_head.pt --frame 6 --steps 190 --out $OUT/videos/fold_198
```

Use the same `SIM_ARM` and `SIM_SELFCOLLIDE` as for the data. `compose_video.py` writes four videos:

| File | Shows |
| --- | --- |
| `_wm_vs_sim.mp4` | The world model alone above the simulator, four cameras |
| `_scene_vs_sim.mp4` | Static background + kinematic arm model + world model above the simulator |
| `_layers.mp4` | One camera, every layer side by side |
| `_three.mp4` | One camera (`--cam`): world model prediction, complete reconstruction, simulator |

`mujoco_cloth/view_recon.py` checks the autoencoder alone: it encodes and decodes a few frames and puts them under the simulator images.

### What we measured

One workstation GPU, YAM arms, a 25 × 25 cloth.

| Dataset | Episodes × frames | Open-loop latent error, step 1 → last step |
| --- | --- | --- |
| `hold` | 100 × 150 | 0.0001 → 0.0006 to 0.0011 after 100 steps |
| `grasp` | 100 × 150 | 0.0001 → 0.0016 to 0.0026 after 100 steps |
| `fold` | 200 × 200 | 0.0001 to 0.0002 → 0.0012 to 0.0030 after 190 steps |

- `hold` follows the simulator closely for 10 s. On `fold` the first move is right, then the cloth shape drifts and the pattern smears over the 19 s rollout.
- A dynamics step takes 13 ms, decoding to Gaussians 92 ms and rendering one 640 × 480 view 8 ms. MuJoCo takes about 57 ms per 0.1 s frame on one CPU core without cloth self-collision and about 175 ms with it.
- The joint head is off by 0.44° per joint on true latents, which puts the fingertip 5 mm off on average.

[videos/mujoco_yam_hold_ep90.mp4](videos/mujoco_yam_hold_ep90.mp4) is the `_three.mp4` video of held-out `hold` episode 90, a 140-step rollout.

[Handoff notes](HANDOFF.md) cover the method, measured results, benchmark protocol, file map and next experiments.

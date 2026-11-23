#!/bin/bash
# Simulate a cloth dataset in MuJoCo, fit the 3D World Simulator to it and write open-loop rollout videos.
#
#   WS_ROOT=~/3d-world-simulator mujoco_cloth/run_3dws.sh fold      # or hold, grasp
#
# Settings (environment): SIM_ARM yam (default) or vx300s; OUT work folder (default ./results/mujoco);
# EPISODES, FRAMES, VAL; VAE_EPOCHS, DYN_EPOCHS; WORKERS, RENDER_WORKERS (each render worker holds an episode of images, ~5 GB).
set -e
MODE=${1:?hold, grasp or fold}
: "${WS_ROOT:?set WS_ROOT to your clone of https://github.com/jwang771/3d-world-simulator}"
HERE=$(cd "$(dirname "$0")" && pwd)
export WS_ROOT SIM_ARM=${SIM_ARM:-yam} MUJOCO_GL=${MUJOCO_GL:-egl}
[ "$MODE" = fold ] && export SIM_SELFCOLLIDE=${SIM_SELFCOLLIDE:-auto}   # folds need the cloth to rest on itself
OUT=$(mkdir -p "${OUT:-results/mujoco}" && cd "${OUT:-results/mujoco}" && pwd)
N=simcloth_${MODE}_${SIM_ARM}; D=$OUT/data/$N; R=$OUT/runs/$N
EPISODES=${EPISODES:-200}; FRAMES=${FRAMES:-200}; VAL=${VAL:-20}
mkdir -p "$R" "$OUT/videos"

echo "== simulate and render $N"
python "$HERE/cloth_arms.py" --mode "$MODE" --out "$D" --episodes "$EPISODES" --frames "$FRAMES" --val "$VAL" \
  --workers "${WORKERS:-16}" --render-workers "${RENDER_WORKERS:-5}"
grep -q "\"$N\"" "$WS_ROOT/tasks.py" || python "$HERE/register_task.py" "$D"

cd "$WS_ROOT"
echo "== autoencoder"
python scripts/train_vae.py --recipe minimal --task "$N" --data-root "$OUT/data" --out-dir "$R/vae" \
  --n-epochs "${VAE_EPOCHS:-30}" --iters 300 --val-iters 20 --val-interval 5 --save-interval 5 --phase-a 8 --phase-b 8
echo "== latents"
for split in train val; do
  python scripts/export_latents.py --vae-ckpt "$R/vae/checkpoints/latest.pth" --packed-dir "$D/$split" --out-dir "$R/latents/$split"
done
echo "== dynamics"
E=${DYN_EPOCHS:-80}
python scripts/train_dynamics.py --latent-dir "$R/latents/train" --val-latent-dir "$R/latents/val" --out-dir "$R/dynamics" \
  --n-history 4 --ar-schedule 1 2 3 4 5 6 --ar-schedule-epochs $((E / 8)) $((E / 4)) $((3 * E / 8)) $((E / 2)) $((5 * E / 8)) "$E" \
  --iters 300 --val-iters 20 --object-alpha 20 --save-interval 5

echo "== kinematic arm model, static background, joint head"
python "$HERE/fit_rig.py" --states "$D/states" --out "$R/rig.pt"
python "$HERE/joint_head.py" --latents "$R/latents" --packed "$D/packed" --out "$R/joint_head.pt"

echo "== open-loop rollout videos of held-out episodes"
for ep in $(ls "$D/val" | sed 's/episode_\([0-9]*\).hdf5/\1/' | sort -n | awk 'NR % 3 == 1'); do
  python scripts/export_release.py --out "$R/task_$ep" --run "$R/vae/checkpoints/latest.pth" "$R/dynamics/checkpoints/latest.pth" \
    "$R/latents/val/episode_${ep}_latents.h5" "$D/val/episode_$ep.hdf5"
  python "$HERE/compose_video.py" --task "$R/task_$ep" --packed "$D/packed/episode_$ep.hdf5" --rig "$R/rig.pt" \
    --joint-head "$R/joint_head.pt" --frame 6 --steps $((FRAMES - 10)) --out "$OUT/videos/${N}_ep$ep"
done
echo "videos in $OUT/videos"

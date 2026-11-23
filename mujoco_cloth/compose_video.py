"""Open-loop rollout videos of a simulated task: the world model next to the simulator, and the whole scene rebuilt from
static background + kinematic arm model + world model next to the simulator.

The world model gets the first frames of a held-out episode and then only the actions. Per frame:

  world model          the Gaussians decoded from the predicted latent (only the workspace box exists for it)
  static background    the fitted floor (fit_rig.py)
  kinematic model      the fitted arm Gaussians, posed with forward kinematics from the joint angles that the joint head
                       (joint_head.py) reads from the predicted latent
  whole scene          background + kinematic model + the world model without its own (soft) arms
  simulator            the simulator's image of that frame

Writes <out>_wm_vs_sim.mp4, <out>_scene_vs_sim.mp4 (four cameras each), <out>_layers.mp4 (one camera, every layer) and
<out>_three.mp4 (one camera: world model, whole scene, simulator).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
WS_ROOT = os.environ.get("WS_ROOT")  # a clone of https://github.com/jwang771/3d-world-simulator
if not WS_ROOT:
    raise SystemExit("set WS_ROOT to your clone of the 3D World Simulator repository")
sys.path.insert(0, WS_ROOT)
import cloth_arms as ca  # noqa: E402
import mujoco  # noqa: E402
from fit_rig import arm_bodies, body_poses, posed  # noqa: E402
from joint_head import JointHead  # noqa: E402
from models.gsplat_render_loss import render_gaussians  # noqa: E402
from models.world_model import WorldModel  # noqa: E402

ARM_Z = 0.012  # world-model Gaussians that are not cloth and lie higher than this above the floor are its arms


def label(im, text):
    im = np.ascontiguousarray(im)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(im, (0, 0), (w + 14, h + 14), (255, 255, 255), -1)
    cv2.putText(im, text, (7, h + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return im


class Video:
    def __init__(self, path, fps):
        self.path, self.fps, self.p = str(path), fps, None

    def write(self, im):
        if self.p is None:
            h, w = im.shape[:2]
            self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", self.path], stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="exported task folder (scripts/export_release.py)")
    p.add_argument("--packed", required=True, help="the packed episode of the task's seed: simulator images and joints")
    p.add_argument("--rig", required=True)
    p.add_argument("--joint-head", required=True)
    p.add_argument("--frame", type=int, default=6)
    p.add_argument("--steps", type=int, default=190)
    p.add_argument("--smooth", type=float, default=1.0, help="exponential smoothing of the joint head (1 = none)")
    p.add_argument("--feather", type=float, default=0.05, help="the world model's floor fades out over this distance at the box faces (m)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cam", type=int, default=0, help="the camera of the one-camera videos (index into the dataset cameras)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda"
    wm = WorldModel(a.task, background=False, arms=False)
    z, act = wm.load_episode(None)
    st = wm.seed(z, act, a.frame)
    cams = wm.episode_cameras(None, frame=0)
    head = JointHead(a.joint_head, dev)
    ck = torch.load(a.rig, map_location=dev, weights_only=False)
    rig, floor = ck["rig"], ck["floor"]
    m, arms = ca.build("fold")
    d = mujoco.MjData(m)
    bodies = arm_bodies(m)
    assert [m.body(b).name for b in bodies] == ck["bodies"]
    mirror = np.array([arm["fingers"][1] for arm in arms])
    # the floor lies a little lower inside the workspace box, so that the world model's floor (with its shadows) is in front there
    inside = ((floor["xyz"][:, :2] > wm.bmin[:2]) & (floor["xyz"][:, :2] < wm.bmax[:2])).all(1)
    floor = dict(floor, xyz=floor["xyz"] - torch.tensor([0, 0, 0.004], device=dev) * inside[:, None])
    zeros = lambda n: torch.zeros(n, 3, 3, device=dev)

    def splat(g, cam, white=False):
        vm, K, W, H, _ = cam
        if g is None:
            return np.full((H, W, 3), 255 if white else 0, np.uint8)
        img, alpha = render_gaussians(g["xyz"], g["scale"], g["opacity"], g["rgb01"], vm[None], K[None], W, H, quats=g["quat"], sh1=g.get("sh1", zeros(len(g["xyz"]))), far_plane=20.0)
        img = img[0] + (1 - alpha[0]) if white else img[0]
        return (img.clamp(0, 1) * 255).byte().cpu().numpy()

    def cat(*gs):
        gs = [dict(g, sh1=g.get("sh1", zeros(len(g["xyz"])))) for g in gs if g is not None]
        return {k: torch.cat([g[k] for g in gs]) for k in ("xyz", "scale", "opacity", "rgb01", "quat", "sh1")}

    f = h5py.File(a.packed, "r")
    sim = lambda c, t: cv2.imdecode(np.frombuffer(f[f"render/camera_{c}/jpg"][t], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    qadr = head.qadr
    vids = {k: Video(f"{a.out}_{k}.mp4", a.fps) for k in ("wm_vs_sim", "scene_vs_sim", "layers", "three")}
    q_s, tip_err, joint_err = None, [], []
    with torch.no_grad():
        for i in range(a.steps):
            t = st.t
            if t >= len(act):
                break
            zp = wm.step(st, act[t])
            g = wm.decode(zp)
            # kinematic model from the predicted latent
            q = head(zp)
            q_s = q if q_s is None else q_s + a.smooth * (q - q_s)
            d.qpos[qadr], d.qpos[mirror] = q_s, -q_s[[6, 13]]
            mujoco.mj_kinematics(m, d)
            tips = np.stack([ca.held_point(d, arm) for arm in arms])
            arm_g = posed(rig, *body_poses(m, d, bodies, dev))
            if t < f["sim/qpos"].shape[0]:
                joint_err.append(np.abs(q_s - f["sim/qpos"][t][qadr])[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean())
                tip_err.append(np.linalg.norm(tips - f["sim/pinch"][t], axis=-1).mean())
            # the world model without its own arms, its floor fading out at the box faces
            wm_g = None
            if g is not None:
                x = g["xyz"]
                keep = ~((g["cls"] != 0) & (x[:, 2] > ARM_Z))
                dface = torch.stack([x[:, 0] - wm.bmin[0], wm.bmax[0] - x[:, 0], x[:, 1] - wm.bmin[1], wm.bmax[1] - x[:, 1]], 1).min(1).values
                w = (dface / a.feather).clamp(0, 1)
                fade = torch.where((g["cls"] != 0) & (x[:, 2] <= ARM_Z), w * w * (3 - 2 * w), torch.ones_like(w))
                wm_g = {k: v[keep] for k, v in dict(g, opacity=g["opacity"] * fade).items()}
            scene = cat(wm_g, arm_g, floor)
            sims = [sim(c[4], t) for c in cams]
            row = lambda ims, text: label(np.concatenate(ims, 1), text)
            wm_row = row([splat(g, c, white=True) for c in cams], "world model, open loop")
            scene_row = row([splat(scene, c) for c in cams], "whole scene: static background + kinematic arm model + world model (open loop)")
            sim_row = row(sims, "simulator")
            vids["wm_vs_sim"].write(np.concatenate([wm_row, sim_row], 0))
            vids["scene_vs_sim"].write(np.concatenate([scene_row, sim_row], 0))
            c = cams[a.cam]
            panels = [label(splat(floor, c), "static background"), label(splat(arm_g, c), "kinematic arm model"), label(splat(g, c, white=True), "world model"), label(splat(scene, c), "whole scene"), label(sims[a.cam], "simulator")]
            vids["layers"].write(np.concatenate(panels, 1))
            three = [label(splat(g, c, white=True), "world model prediction (open loop)"), label(splat(scene, c), "complete reconstruction"), label(sims[a.cam], "simulator (ground truth)")]
            vids["three"].write(np.concatenate(three, 1))
    for v in vids.values():
        v.close()
    print(f"[compose] {a.out}: joint head vs simulator over the rollout: joints {np.degrees(np.mean(joint_err)):.2f} deg, gripper tip {1000 * np.mean(tip_err):.1f} mm (mean), {1000 * np.max(tip_err):.1f} mm (worst frame)")


if __name__ == "__main__":
    main()

"""Joint head for the simulated arms: joint angles and finger opening of both arms, read from a world-model latent.

The kinematic model (fit_rig.py) is posed from this, i.e. from what the world model predicts, never from the commanded action.

    SIM_ARM=yam python joint_head.py --latents runs/x/latents --packed data/x/packed --out runs/x/joint_head.pt
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
WS_ROOT = os.environ.get("WS_ROOT")  # a clone of https://github.com/jwang771/3d-world-simulator
if not WS_ROOT:
    raise SystemExit("set WS_ROOT to your clone of the 3D World Simulator repository")
sys.path.insert(0, WS_ROOT)


class JointNet(nn.Module):
    """Latent (8, nz, ny, nx) -> 14 numbers (per arm six joints and one finger), normalised."""

    def __init__(self, shape, width=64, n_out=14):
        super().__init__()
        c = lambda i, o, s: nn.Sequential(nn.Conv3d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(c(8 + 3, width, 1), c(width, width * 2, 2), c(width * 2, width * 4, 2), c(width * 4, width * 4, 2))
        zs = [-(-s // 2) for s in shape]  # the input is average-pooled by 2 first
        feat = int(np.prod([-(-(-(-(-(-s // 2)) // 2)) // 2) for s in zs]))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 4 * feat, 512), nn.SiLU(), nn.Linear(512, n_out))
        grid = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in zs], indexing="ij"))
        self.register_buffer("grid", grid[None])

    def forward(self, z_pooled):
        return self.head(self.net(torch.cat([z_pooled, self.grid.expand(len(z_pooled), -1, -1, -1, -1)], 1)))


class JointHead:
    """latent -> arm joint positions as the simulator's qpos entries: ``head(z) -> (values (14,), qpos addresses (14,))``."""

    def __init__(self, path, device="cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = JointNet(ck["shape"]).to(device).eval()
        self.net.load_state_dict(ck["state"])
        self.mean, self.std, self.qadr = torch.tensor(ck["mean"], device=device), torch.tensor(ck["std"], device=device), np.asarray(ck["qadr"])

    @torch.no_grad()
    def __call__(self, z):
        z = torch.as_tensor(z, dtype=torch.float32, device=self.mean.device)[None]
        return (self.net(F.avg_pool3d(z, 2, ceil_mode=True))[0] * self.std + self.mean).cpu().numpy()


def joint_addresses():
    """qpos addresses of [arm 0 joints (6), arm 0 finger, arm 1 joints (6), arm 1 finger] and of the mirrored second fingers."""
    import cloth_arms as ca

    m, arms = ca.build("fold")
    return np.concatenate([np.r_[arm["qadr"], arm["fingers"][0]] for arm in arms]), np.array([arm["fingers"][1] for arm in arms]), m


def load_split(latent_dir, packed_dir, qadr, stride):
    from models.world_model import read_dense_z

    Z, Q = [], []
    for f in sorted(Path(latent_dir).glob("episode_*_latents.h5")):
        ep = int(f.name.split("_")[1])
        with h5py.File(f, "r") as h:
            z = read_dense_z(h)[::stride]
        with h5py.File(Path(packed_dir) / f"episode_{ep}.hdf5", "r") as h:
            q = h["sim/qpos"][:][::stride][:, qadr]
        Z.append(F.avg_pool3d(torch.from_numpy(z), 2, ceil_mode=True).half())
        Q.append(torch.from_numpy(q).float())
    return torch.cat(Z), torch.cat(Q)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="folder with train/ and val/ latent episodes")
    p.add_argument("--packed", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.05, help="noise added to the latents in training (rollout latents are not exact)")
    a = p.parse_args()
    import mujoco

    qadr, mirror, m = joint_addresses()
    Zt, Qt = load_split(Path(a.latents) / "train", a.packed, qadr, a.stride)
    Zv, Qv = load_split(Path(a.latents) / "val", a.packed, qadr, a.stride)
    mean, std = Qt.mean(0), Qt.std(0).clamp(min=1e-4)
    print(f"train {tuple(Zt.shape)}, val {tuple(Zv.shape)}, latent std {Zt[:200].float().std():.3f}", flush=True)
    dev = "cuda"
    shape = [s * 2 for s in Zt.shape[2:]]
    net = JointNet(shape).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    steps = a.epochs * (len(Zt) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    mean_d, std_d = mean.to(dev), std.to(dev)

    def evaluate():
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Zv), 128):
                out.append(net(Zv[i : i + 128].to(dev).float()) * std_d + mean_d)
        net.train()
        pred = torch.cat(out).cpu()
        err = (pred - Qv).abs()
        return pred, err

    for ep in range(a.epochs):
        perm = torch.randperm(len(Zt))
        tot = 0.0
        for i in range(0, len(perm) - 63, 64):
            idx = perm[i : i + 64]
            z = Zt[idx].to(dev).float()
            z = z + a.noise * torch.randn_like(z)
            loss = F.smooth_l1_loss(net(z), (Qt[idx].to(dev) - mean_d) / std_d, beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            _, err = evaluate()
            joints = err[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
            print(f"[ep {ep}] train {tot / (len(perm) // 64):.4f}  val joint error {np.degrees(joints.mean()):.2f} deg (max joint mean {np.degrees(joints.mean(0).max()):.2f}), finger {1000 * err[:, [6, 13]].mean():.2f} mm", flush=True)
    torch.save(dict(state=net.state_dict(), shape=shape, mean=mean.numpy(), std=std.numpy(), qadr=qadr, mirror=mirror), a.out)
    # how far the gripper tip is off with the predicted joints
    import cloth_arms as ca

    _, arms = ca.build("fold")
    d = mujoco.MjData(m)
    pred, _ = evaluate()
    tip = []
    for k in range(0, len(pred), 7):
        pts = []
        for q in (pred[k].numpy(), Qv[k].numpy()):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            pts.append(np.stack([ca.held_point(d, arm) for arm in arms]))
        tip.append(np.linalg.norm(pts[0] - pts[1], axis=-1))
    tip = np.concatenate(tip) * 1000
    print(f"gripper tip error with predicted joints: mean {tip.mean():.1f} mm, median {np.median(tip):.1f}, 95% {np.percentile(tip, 95):.1f}, max {tip.max():.1f}", flush=True)


if __name__ == "__main__":
    main()

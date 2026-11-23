"""Fit the Gaussian kinematic model of the simulated arms and the static background of the simulated scene.

  kinematic model     Gaussians attached to the links of both arms (initialised on the link meshes), fitted to simulator renders
                      of the arms at joint configurations taken from a dataset, from the dataset cameras and random ones. Posed
                      with forward kinematics from joint angles (`posed`).
  static background   Gaussians on the floor, fitted to renders of the empty scene. Everything that never moves.

    SIM_ARM=yam python mujoco_cloth/fit_rig.py --states <dataset>/states --out <dir>/rig.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloth_arms as ca  # noqa: E402
import mujoco  # noqa: E402
from gsplat import rasterization  # noqa: E402

FAR = dict(x=0.0, y=30.0, yaw=0.0)  # where the cloth is parked so that it is not in the fitting images


def camera_pose(scene):
    """OpenCV camera-to-world (4, 4) of the camera MuJoCo just rendered with."""
    cams = scene.camera
    pos = (np.array(cams[0].pos) + np.array(cams[1].pos)) / 2
    f, up = np.array(cams[0].forward), np.array(cams[0].up)
    x = np.cross(f, up)
    E = np.eye(4)
    E[:3, 0], E[:3, 1], E[:3, 2], E[:3, 3] = x, -up, f, pos
    return E


def intrinsics(W, H, fovy):
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)


def random_camera(rng, look):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = rng.uniform(0.45, 1.3), rng.uniform(0, 360), -rng.uniform(12, 80)
    return cam


def arm_bodies(m):
    """Ids of the bodies of both arms, in model order."""
    return [b for b in range(m.nbody) if m.body(b).name.startswith(("arm0_", "arm1_"))]


def sample_links(m, spacing, rng):
    """Points on the visual meshes of every arm link, in the link frame: body index (in arm_bodies order), xyz, normal, rgb."""
    bodies = arm_bodies(m)
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in bodies or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        mesh = int(m.geom_dataid[g])
        V = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        Fc = m.mesh_face[m.mesh_faceadr[mesh] : m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
        a, bb, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
        nrm = np.cross(bb - a, c - a)
        area = np.linalg.norm(nrm, axis=1) / 2
        n = max(50, int(area.sum() / spacing**2))
        tri = rng.choice(len(Fc), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        p = a[tri] + (bb[tri] - a[tri]) * u[:, None] + (c[tri] - a[tri]) * v[:, None]
        nn = nrm[tri] / np.maximum(np.linalg.norm(nrm[tri], axis=1, keepdims=True), 1e-12)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R = R.reshape(3, 3)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        out.append((np.full(n, bodies.index(b)), p @ R.T + m.geom_pos[g], nn @ R.T, np.tile(rgba[:3], (n, 1))))
    return [np.concatenate(x) for x in zip(*out)]


def normal_quat(n):
    """Quaternions (wxyz) that turn the z axis onto each normal."""
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    w = 1 + n @ z
    q = np.concatenate([w[:, None], v], 1)
    q[w < 1e-6] = [0, 1, 0, 0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def posed(rig, xpos, xquat):
    """World-frame Gaussians of the kinematic model. xpos (B, 3), xquat (B, 4): pose of every arm body (arm_bodies order)."""
    b = rig["body"]
    R = quat_to_mat(xquat)[b]
    return dict(
        xyz=(R @ rig["xyz"][..., None])[..., 0] + xpos[b], quat=quat_mul(xquat[b], torch.nn.functional.normalize(rig["quat"], dim=-1)),
        scale=rig["log_scale"].exp(), opacity=torch.sigmoid(rig["opacity"]), rgb01=torch.sigmoid(rig["rgb"]),
    )


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def body_poses(m, d, bodies, device):
    return torch.tensor(d.xpos[bodies], dtype=torch.float32, device=device), torch.tensor(d.xquat[bodies], dtype=torch.float32, device=device)


def splat(g, E, K, W, H, bg):
    vm = torch.linalg.inv(E)[None]
    sh1 = g.get("sh1")  # view-dependent colour (degree-1 spherical harmonics), same convention as models/gsplat_render_loss.py
    sh1 = torch.zeros(len(g["xyz"]), 3, 3, device=g["xyz"].device) if sh1 is None else sh1
    colors = torch.cat([((g["rgb01"] - 0.5) / 0.28209479177387814)[:, None], sh1], 1)
    img, alpha, _ = rasterization(g["xyz"], g["quat"], g["scale"], g["opacity"], colors, vm, K[None], W, H, sh_degree=1, near_plane=0.02, far_plane=20.0, rasterize_mode="antialiased")
    return img[0] + (1 - alpha[0]) * bg, alpha[0, ..., 0]


def empty_scene():
    """The scene without arms and cloth: only what never moves."""
    cams = ""
    spec = mujoco.MjSpec.from_string(ca.SCENE.format(cams=cams, x=FAR["x"], y=FAR["y"], yaw=0.0))
    return spec.compile()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", required=True, help="<dataset>/states of a simulated dataset: where the joint configurations come from")
    p.add_argument("--out", required=True)
    p.add_argument("--images", type=int, default=260)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--spacing", type=float, default=0.002, help="one Gaussian per this-squared of link surface (m)")
    p.add_argument("--floor", type=float, default=1.5, help="half size of the floor (m)")
    p.add_argument("--floor-spacing", type=float, default=0.005)
    a = p.parse_args()
    dev, rng = "cuda", np.random.default_rng(0)
    W, H = ca.W, ca.H
    K = torch.tensor(intrinsics(W, H, ca.FOVY), device=dev)

    # ---- kinematic model
    m, arms = ca.build("fold", **FAR)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    bodies = arm_bodies(m)
    arm_geom = np.isin(m.geom_bodyid, bodies)
    files = sorted(Path(a.states).glob("episode_*.npz"))
    data = []
    for i in range(a.images):
        s = np.load(files[int(rng.integers(len(files)))])
        d.qpos[:] = s["qpos"][int(rng.integers(len(s["qpos"])))]
        mujoco.mj_forward(m, d)
        cam = int(rng.integers(len(ca.CAMS))) if i % 3 == 0 else random_camera(rng, [rng.uniform(-0.3, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(0.02, 0.2)])
        r.update_scene(d, camera=cam)
        rgb = r.render().copy()
        E = camera_pose(r.scene)
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & arm_geom[np.clip(seg[..., 0], 0, m.ngeom - 1)]
        data.append((torch.tensor(rgb, device=dev).float() / 255, torch.tensor(mask, device=dev).float(), torch.tensor(E, dtype=torch.float32, device=dev), *body_poses(m, d, bodies, dev)))
    body, xyz, nrm, rgb = sample_links(m, a.spacing, rng)
    print(f"kinematic model: {len(xyz):,} Gaussians on {len(bodies)} links, {len(data)} images", flush=True)
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    flat = np.log(np.tile([a.spacing * 0.9, a.spacing * 0.9, a.spacing * 0.25], (len(xyz), 1)))
    rig = dict(body=torch.tensor(body, device=dev), xyz=t(xyz), quat=t(normal_quat(nrm)), log_scale=t(flat), opacity=t(np.full(len(xyz), 2.5)), rgb=torch.logit(t(rgb).clamp(0.03, 0.97)))
    params = {k: rig[k].requires_grad_() for k in ("xyz", "quat", "log_scale", "opacity", "rgb")}
    lr = dict(xyz=2e-5, quat=1e-3, log_scale=3e-3, opacity=2e-2, rgb=1e-2)
    opt = torch.optim.Adam([dict(params=[v], lr=lr[k]) for k, v in params.items()])
    xyz0 = rig["xyz"].detach().clone()
    for it in range(a.iters):
        img, mask, E, xpos, xquat = data[int(rng.integers(len(data)))]
        bg = torch.rand(3, device=dev)
        out, alpha = splat(posed(rig, xpos, xquat), E, K, W, H, bg)
        target = img * mask[..., None] + bg * (1 - mask[..., None])
        loss = (out - target).abs().mean() + 0.3 * (alpha - mask).abs().mean() + 10.0 * (rig["xyz"] - xyz0).pow(2).sum(-1).mean() / a.spacing**2 * 1e-3
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            rig["log_scale"].clamp_(np.log(2e-4), np.log(0.006))
        if it % 500 == 0 or it == a.iters - 1:
            print(f"  [arms {it}] loss {loss.item():.4f} l1 {(out - target).abs().mean().item():.4f}", flush=True)

    # ---- static background: the floor
    me = empty_scene()
    de = mujoco.MjData(me)
    mujoco.mj_forward(me, de)
    re = mujoco.Renderer(me, H, W)
    fdata = []
    for i in range(300):
        re.update_scene(de, camera=random_camera(rng, [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0]))
        if i % 4 == 0:  # also from far away, so that all of the floor is seen
            cam = random_camera(rng, [0, 0, 0])
            cam.distance = rng.uniform(1.5, 3.0)
            re.update_scene(de, camera=cam)
        fdata.append((torch.tensor(re.render().copy(), device=dev).float() / 255, torch.tensor(camera_pose(re.scene), dtype=torch.float32, device=dev)))
    gx = np.arange(-a.floor, a.floor, a.floor_spacing) + a.floor_spacing / 2
    fx, fy = np.meshgrid(gx, gx, indexing="ij")
    n = fx.size
    floor = dict(
        xyz=t(np.stack([fx.ravel(), fy.ravel(), np.zeros(n)], 1)), quat=t(np.tile([1.0, 0, 0, 0], (n, 1))),
        scale=t(np.tile([a.floor_spacing * 0.75, a.floor_spacing * 0.75, 2e-4], (n, 1))), opacity=torch.full((n,), 0.97, device=dev),
    )
    frgb = torch.zeros(n, 3, device=dev, requires_grad=True)
    fsh = torch.zeros(n, 3, 3, device=dev, requires_grad=True)  # the simulator's headlight makes the floor brighter from above
    opt = torch.optim.Adam([frgb, fsh], lr=0.03)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.01 ** (1 / 4000))
    black = torch.zeros(3, device=dev)
    print(f"static background: {n:,} Gaussians, {len(fdata)} images", flush=True)
    for it in range(4000):
        img, E = fdata[int(rng.integers(len(fdata)))]
        out, _ = splat(dict(floor, rgb01=torch.sigmoid(frgb), sh1=fsh), E, K, W, H, black)
        loss = (out - img).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == 3999:
            print(f"  [floor {it}] l1 {loss.item():.4f}", flush=True)

    out = dict(
        arm=os.environ.get("SIM_ARM", "vx300s"), bodies=[m.body(b).name for b in bodies],
        rig={k: v.detach().cpu() for k, v in rig.items()}, floor=dict({k: v.cpu() for k, v in floor.items()}, rgb01=torch.sigmoid(frgb).detach().cpu(), sh1=fsh.detach().cpu()),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)

    # ---- a check image: simulator above, kinematic model + background below, three held-out configurations
    import cv2

    rows = []
    g_floor = {k: v.to(dev) for k, v in out["floor"].items()}
    with torch.no_grad():
        for i in range(3):
            s = np.load(files[-1 - i])
            d.qpos[:] = s["qpos"][60 + 40 * i]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=i)
            sim = r.render().copy()
            g = posed(rig, *body_poses(m, d, bodies, dev))
            g["sh1"] = torch.zeros(len(g["xyz"]), 3, 3, device=dev)
            both = {k: torch.cat([g[k], g_floor[k]]) for k in g}
            img, _ = splat(both, torch.tensor(camera_pose(r.scene), dtype=torch.float32, device=dev), K, W, H, black)
            rows.append(np.concatenate([sim, (img.clamp(0, 1) * 255).byte().cpu().numpy()], 0))
    cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), np.concatenate(rows, 1)[..., ::-1])
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()

"""Simulated cloth manipulation with two robot arms in MuJoCo, written as packed episodes for the 3D World Simulator.

A square cloth (MuJoCo flex) lies on a table between two arms (ViperX 300 S, or I2RT YAM with SIM_ARM=yam). Three datasets:

  hold    each arm holds one corner of the cloth for the whole episode and the two grippers move together along a short path
          (lift, shift, fold over, lay down).
  grasp   the cloth starts flat at a random pose; each arm reaches for a point of the cloth, closes its gripper on it, carries
          it somewhere, opens the gripper and retreats.
  fold    longer episodes made of several moves in a row at different speeds: folds with both arms (an edge onto the opposite
          edge or the middle), folds with one arm (a corner or edge point onto its mirror point), lifting and carrying with both
          arms, carrying one point through a few waypoints, dragging along the table. See `Script`.

A grasp is a constraint that ties one cloth vertex to the gripper's pinch point while the gripper is closed. The arms are driven
by their position actuators from an inverse-kinematics solution of the commanded gripper position, which is the action:

    action (14,) = per arm [commanded x y z of the pinch point, 0 0 0, gripper closed (1) or open (0)]

Every episode is rendered from fixed virtual cameras (colour, depth, cloth mask). The workspace box and the voxel grid are
fitted to where the cloth and the grippers actually go in the dataset, and written to `<out>/grid.json`.

    MUJOCO_GL=egl python mujoco_cloth/cloth_arms.py --mode hold  --out data/simcloth_hold  --episodes 100 --workers 10
    MUJOCO_GL=egl python mujoco_cloth/cloth_arms.py --mode grasp --out data/simcloth_grasp --episodes 100 --workers 10
    SIM_ARM=yam SIM_SELFCOLLIDE=auto MUJOCO_GL=egl python mujoco_cloth/cloth_arms.py --mode fold --out data/simcloth_fold_yam \
        --episodes 200 --frames 200 --val 20 --workers 22 --render-workers 5

SIM_ARM=yam uses two I2RT YAM arms instead of the ViperX (default vx300s).
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# robot models: git clone https://github.com/google-deepmind/mujoco_menagerie next to this file, or point MUJOCO_MENAGERIE at a clone
MENAGERIE = Path(os.environ.get("MUJOCO_MENAGERIE", str(Path(__file__).resolve().parent / "mujoco_menagerie")))
ROBOTS = {  # per arm model: its file, names, the pinch point in the gripper body, which axis of the pinch site points out of the gripper
    "vx300s": dict(
        xml="trossen_vx300s/vx300s.xml", joints=("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"), gripper="gripper",
        fingers=("left_finger", "right_finger"), site="pinch", body="gripper_link", pinch=(0.1, 0, 0), axis=0, home=(0, -0.96, 1.16, 0, -0.3, 0),
        open=0.057, closed=0.021, base_x=0.45,
    ),
    "yam": dict(
        xml="i2rt_yam/yam.xml", joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"), gripper="gripper",
        # the YAM fingers are wedges whose tips sit 4 cm to one side of the grasp site (its x axis): the cloth is held at the tips, and
        # the gripper is rolled so that side points down ("tip_axis")
        fingers=("left_finger", "right_finger"), site="grasp_site", body="link_6", pinch=(0, -0.04, 0.1397), tip=(0.04, 0, 0.005), axis=2, tip_axis=0,
        home=(0, 1.047, 1.047, 0, 0, 0), tilt=0.9, open=0.037, closed=0.0, base_x=0.42, base_z=0.0105,  # its base mesh reaches 1 cm below its origin
        reach_z=(0.016, 0.06, 0.15, 0.24), reach_r=(0.57, 0.57, 0.52, 0.41),  # how far from its base the held point can go, by height
    ),
}
ROBOT = ROBOTS[os.environ.get("SIM_ARM", "vx300s")]
N, SPACING = 25, 0.0125  # cloth vertices per side and their spacing: a 0.30 m square
H, W, FOVY = 480, 640, 45.0
FPS, DT = 10, 0.002
ARM_X = ROBOT["base_x"]  # the arm bases sit at x = -ARM_X (arm 0) and +ARM_X (arm 1), facing each other
CAMS = [(0.0, -0.70, 0.52), (0.50, 0.46, 0.50), (-0.50, 0.46, 0.50), (0.0, 0.04, 0.90)]  # positions; all look at LOOK
LOOK = (0.0, 0.0, 0.06)
GRASP_STRIDE = 3  # in the grasp dataset every third vertex per side can be grasped
OPEN, CLOSED = ROBOT["open"], ROBOT["closed"]  # finger actuator targets
TABLE_Z = 0.016  # height of the pinch point when it is at the cloth on the table

SELF = os.environ.get("SIM_SELFCOLLIDE", "none")  # "auto" lets the cloth rest on itself (needed for folds; 3x slower)
SCENE = f"""
<mujoco>
  <option timestep="{DT}" integrator="implicitfast" solver="CG" tolerance="1e-6"/>
  <size memory="600M"/>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096"/><headlight ambient="0.42 0.42 0.42" diffuse="0.40 0.40 0.40" specular="0 0 0"/></visual>
  <asset>
    <texture name="table" type="2d" builtin="checker" rgb1="0.80 0.79 0.76" rgb2="0.74 0.73 0.70" width="256" height="256"/>
    <material name="table" texture="table" texrepeat="6 6" reflectance="0"/>
    <texture name="cloth" type="2d" builtin="checker" rgb1="0.20 0.62 0.36" rgb2="0.93 0.86 0.30" width="256" height="256"/>
    <material name="cloth" texture="cloth" texrepeat="4 4" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.5" dir="-0.2 0.2 -1" diffuse="0.32 0.32 0.32" castshadow="true"/>
    <geom name="table" type="plane" size="1.5 1.5 0.1" material="table" friction="0.6"/>
    <flexcomp name="cloth" type="grid" count="{N} {N} 1" spacing="{SPACING} {SPACING} {SPACING}" pos="{{x}} {{y}} 0.01" euler="0 0 {{yaw}}"
              radius="0.004" mass="0.06" dim="2" material="cloth">
      <contact selfcollide="{SELF}" condim="3" solref="0.004 1" friction="0.6"/>
      <edge equality="true" damping="0.002"/>
    </flexcomp>
    {{cams}}
  </worldbody>
</mujoco>
"""


def look_at(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.95 else np.array([0, 1.0, 0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    return x, np.cross(x, f)


def vertex(i, j):
    return i * N + j  # flexcomp grid order: i along x, j along y


def graspable(mode):
    """Cloth vertices each arm can be tied to: -> [vertices of arm 0, vertices of arm 1]."""
    if mode == "hold":  # the two corners of the near edge
        return [[vertex(0, 0)], [vertex(N - 1, 0)]]
    ids = [vertex(i, j) for i in range(0, N, GRASP_STRIDE) for j in range(0, N, GRASP_STRIDE)]
    return [ids, ids]


def build(mode, x=0.0, y=0.0, yaw=0.0):
    """-> model and, per arm, a dict with its joint, actuator, site and constraint ids."""
    cams = ""
    for i, pos in enumerate(CAMS):
        ax, ay = look_at(pos, LOOK)
        cams += f'<camera name="cam{i}" pos="{pos[0]} {pos[1]} {pos[2]}" xyaxes="{ax[0]} {ax[1]} {ax[2]} {ay[0]} {ay[1]} {ay[2]}" fovy="{FOVY}"/>\n'
    spec = mujoco.MjSpec.from_string(SCENE.format(cams=cams, x=x, y=y, yaw=np.degrees(yaw)))
    spec.compiler.degree = True
    names = ("arm0", "arm1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, bx, byaw in ((names[0], -ARM_X, 0.0), (names[1], ARM_X, np.pi)):
            arm = mujoco.MjSpec.from_file(str(MENAGERIE / ROBOT["xml"]))
            for k in list(arm.keys):
                arm.delete(k)
            for light in list(arm.lights):
                arm.delete(light)
            for g in arm.geoms:  # the arms are drawn but do not collide: a grasp is a constraint, not finger contact
                g.contype, g.conaffinity = 0, 0
            for b in arm.bodies[1:]:
                b.gravcomp = 1
            frame = spec.worldbody.add_frame(pos=[bx, 0, ROBOT.get("base_z", 0.0)], quat=[np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)])
            spec.attach(arm, prefix=name + "_", frame=frame)
    cand = graspable(mode)
    for a, name in enumerate(names):
        for v in cand[a]:
            e = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY, name=f"{name}_v{v}", name1=f"{name}_{ROBOT['body']}", name2=f"cloth_{v}", active=False)
            e.data[0:3] = ROBOT["pinch"]  # the pinch point in the gripper body
    m = spec.compile()
    arms = []
    for a, name in enumerate(names):
        joints = [m.joint(f"{name}_{j}") for j in ROBOT["joints"]]
        eq = {v: m.equality(f"{name}_v{v}").id for v in cand[a]}
        for i in eq.values():
            m.eq_data[i, 3:6] = 0  # ... is tied to the vertex itself
        arms.append(
            dict(
                qadr=np.array([j.qposadr[0] for j in joints]),
                dof=np.array([j.dofadr[0] for j in joints]),
                lo=np.array([j.range[0] for j in joints]),
                hi=np.array([j.range[1] for j in joints]),
                act=np.array([m.actuator(f"{name}_{j}").id for j in ROBOT["joints"]]),
                grip=m.actuator(f"{name}_{ROBOT['gripper']}").id,
                fingers=[m.joint(f"{name}_{j}").qposadr[0] for j in ROBOT["fingers"]],
                site=m.site(f"{name}_{ROBOT['site']}").id,
                base=np.array([-ARM_X if a == 0 else ARM_X, 0.0, 0.0]),
                eq=eq,
            )
        )
    return m, arms


def solve_ik(m, d, arm, target, q0, iters=40):
    """Joint angles that put the pinch point at ``target`` with the gripper pointing down and away from its base."""
    out = target - arm["base"]
    out[2] = 0
    out /= max(np.linalg.norm(out), 1e-6)
    tilt = ROBOT.get("tilt", 0.45)
    direction = np.array([tilt * out[0], tilt * out[1], -1.0])
    direction /= np.linalg.norm(direction)
    tip = np.asarray(ROBOT.get("tip", (0, 0, 0)), float)  # the held point, in the site frame
    side = np.array([0, 0, -1.0]) - direction * -direction[2]  # the most downward direction square to the approach
    side /= np.linalg.norm(side)
    q = q0.copy()
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[arm["qadr"]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jp, jr, arm["site"])
        Rs = d.site_xmat[arm["site"]].reshape(3, 3)
        rot = np.cross(Rs[:, ROBOT["axis"]], direction)
        if "tip_axis" in ROBOT:  # also fix the roll about the approach direction
            rot = rot + np.cross(Rs[:, ROBOT["tip_axis"]], side)
        off = Rs @ tip
        err = np.r_[target - (d.site_xpos[arm["site"]] + off), 0.2 * rot]
        skew = np.array([[0, -off[2], off[1]], [off[2], 0, -off[0]], [-off[1], off[0], 0]])
        J = np.r_[(jp - skew @ jr)[:, arm["dof"]], 0.2 * jr[:, arm["dof"]]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + np.clip(dq, -0.3, 0.3), arm["lo"], arm["hi"])
    return q


def held_point(d, arm):
    """Where the gripper holds the cloth, in the world."""
    return d.site_xpos[arm["site"]] + d.site_xmat[arm["site"]].reshape(3, 3) @ np.asarray(ROBOT.get("tip", (0, 0, 0)), float)


def ease(a, b, n):
    """n points from a to b (excluding a), cosine-eased."""
    u = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
    return a[None] * (1 - u[:, None]) + b[None] * u[:, None]


def plan_hold(rng, start, frames):
    """Both grippers follow one shared short path with small individual offsets. -> targets (frames, 2, 3), closed (frames, 2)."""
    centre, half = start.mean(0), (start[1] - start[0]) / 2
    keys = [(centre.copy(), 1.0)]
    for i in range(int(rng.integers(2, 4))):
        c = centre + np.array([rng.uniform(-0.07, 0.07), rng.uniform(-0.03, 0.22), 0.0])
        c[2] = rng.uniform(0.06, 0.22)
        keys.append((c, rng.uniform(0.65, 1.0)))  # the second number scales the distance between the grippers (slack)
    if rng.random() < 0.6:  # lay the cloth back down
        c = keys[-1][0].copy()
        c[2] = TABLE_Z + 0.005
        keys.append((c, keys[-1][1]))
    pts = [np.stack([c - half * s, c + half * s]) + (rng.uniform(-0.015, 0.015, (2, 3)) if i else 0) for i, (c, s) in enumerate(keys)]
    for p in pts:
        p[:, 2] = np.maximum(p[:, 2], TABLE_Z)
    n = (frames - 10) // (len(pts) - 1)
    path = [np.repeat(pts[0][None], 5, 0)] + [ease(pts[i].reshape(-1), pts[i + 1].reshape(-1), n).reshape(n, 2, 3) for i in range(len(pts) - 1)]
    path = np.concatenate(path)
    path = np.concatenate([path, np.repeat(path[-1:], frames - len(path), 0)])
    return path, np.ones((frames, 2))


REACH_Z, REACH_R = ROBOT.get("reach_z", (0.0, 0.3)), ROBOT.get("reach_r", (0.62, 0.62))
GRID = list(range(0, N, GRASP_STRIDE))  # graspable rows / columns of the cloth
EDGE = [(i, j) for i in GRID for j in GRID if i in (0, N - 1) or j in (0, N - 1)]


def reach_clip(p, base):
    """The closest point to ``p`` the arm standing at ``base`` can hold."""
    p = np.array(p, float)
    p[2] = np.clip(p[2], TABLE_Z, REACH_Z[-1])
    v = p[:2] - base[:2]
    r, rmax = np.linalg.norm(v), np.interp(p[2], REACH_Z, REACH_R)
    if r > rmax:
        p[:2] = base[:2] + v * rmax / r
    return p


def ease_u(n):
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)


class Script:
    """A random sequence of moves for both arms, planned on the cloth as it is when each move starts.

    ``cloth()`` returns the current vertex positions (N * N, 3). Iterating yields, per frame, the commanded positions (2, 3), the
    gripper closed flags (2,) and a list of events (arm, "attach", vertex) or (arm, "detach").
    """

    def __init__(self, rng, cloth, bases):
        self.rng, self.cloth, self.bases = rng, cloth, bases
        self.home = [np.array([s * 0.24, rng.uniform(-0.05, 0.05), 0.17]) for s in (-1, 1)]
        self.cur = [h.copy() for h in self.home]

    # ---- what can be grasped
    def top(self, xy, P):
        near = np.linalg.norm(P[:, :2] - xy, axis=1) < 0.02
        return P[near, 2].max() if near.any() else 0.0

    def ok(self, a, ij, P, own_side=False):
        """Vertex ij is the top layer of the cloth where it lies, and arm a reaches it."""
        p = P[vertex(*ij)]
        if self.top(p[:2], P) > p[2] + 0.004 or abs(p[0]) > 0.24 or abs(p[1]) > 0.24:
            return False
        if own_side and (1 if a else -1) * p[0] < -0.02:
            return False
        return self.reaches(a, [p[0], p[1], TABLE_Z])

    def reaches(self, a, p):
        return np.linalg.norm(reach_clip(p, self.bases[a]) - np.asarray(p, float)) < 0.004

    # ---- one arm: reach, grasp, follow a path, let go, go home
    def pick(self, a, ij, path, speed):
        """``path(grab)`` -> list of (f, frames): f(u) is the commanded position for u in 0..1; the last one ends where it lets go."""
        n = lambda frames: max(2, int(round(frames / speed)))
        v, base = vertex(*ij), self.bases[a]
        at = lambda: self.cloth()[v]
        start = self.cur[a].copy()
        for u in ease_u(n(18)):
            yield start + (at() + [0, 0, 0.09] - start) * u, 0.0, None
        for u in ease_u(n(11)):
            p = at()
            yield np.array([p[0], p[1], 0.09 + p[2] + (max(p[2] + 0.008, TABLE_Z) - 0.09 - p[2]) * u]), 0.0, None
        p = at()
        grab = np.array([p[0], p[1], max(p[2] + 0.008, TABLE_Z)])
        for i in range(3):
            yield grab, 1.0, (a, "attach", v) if i == 2 else None
        end = grab
        for f, frames in path(grab):
            for u in ease_u(n(frames)):
                end = reach_clip(f(u), base)
                yield end, 1.0, None
        for i in range(3):
            yield end, 0.0, (a, "detach") if i == 0 else None
        up = reach_clip(end + [0, 0, 0.08], base)
        for u in ease_u(n(9)):
            yield end + (up - end) * u, 0.0, None
        for u in ease_u(n(12)):
            yield up + (self.home[a] - up) * u, 0.0, None

    @staticmethod
    def line(p, q):
        return lambda u: p + (q - p) * u

    def fold_path(self, dst_ij, lift):
        """Carry the grasped point in an arc onto where vertex dst_ij lies now, and let go just above it."""

        def path(grab):
            P = self.cloth()
            d = P[vertex(*dst_ij)].copy()
            d[2] = self.top(d[:2], P) + 0.02
            dist = np.linalg.norm(d[:2] - grab[:2])
            h = min(0.5 * dist * lift, 0.21 - max(grab[2], d[2]))
            return [(lambda u: grab + (d - grab) * u + np.array([0, 0, h * np.sin(np.pi * u)]), max(14, dist / 0.011))]

        return path

    def way_path(self, offsets, drop):
        """Through grab + each offset, then down by ``drop`` (0 keeps the height, 1 goes down to the table) before letting go."""

        def path(grab):
            out, p = [], grab
            for o in offsets:
                q = np.clip(grab + o, [-0.2, -0.2, TABLE_Z + 0.02], [0.2, 0.2, 0.23])
                out.append((self.line(p, q), max(10, np.linalg.norm(q - p) / 0.011)))
                p = q
            if drop > 0.05:
                q = p.copy()
                q[2] = p[2] + (self.top(p[:2], self.cloth()) + 0.03 - p[2]) * drop
                out.append((self.line(p, q), max(8, (p[2] - q[2]) / 0.011)))
            return out

        return path

    # ---- the moves: each returns {arm: generator} or None when the cloth does not allow it now
    def by_x(self, ij0, ij1, P):
        """Give the vertex further to the left to arm 0."""
        return (ij0, ij1) if P[vertex(*ij0), 0] <= P[vertex(*ij1), 0] else (ij1, ij0)

    def edge_pairs(self):
        """Two points on one edge of the cloth and, for folds, the points they map to on the opposite edge / the middle."""
        r = self.rng
        lo, hi = ((0, N - 1), (GRASP_STRIDE, N - 1 - GRASP_STRIDE))[int(r.random() < 0.3)]
        row = int(r.choice([0, N - 1]))
        dst = N - 1 - row if r.random() < 0.65 else N // 2
        if r.random() < 0.5:  # along j
            return ((lo, row), (hi, row)), ((lo, dst), (hi, dst))
        return ((row, lo), (row, hi)), ((dst, lo), (dst, hi))

    def bi_fold(self, speed):
        P = self.cloth()
        for _ in range(12):
            src, dst = self.edge_pairs()
            s0, s1 = self.by_x(*src, P)
            d0, d1 = dst[src.index(s0)], dst[src.index(s1)]
            if self.ok(0, s0, P) and self.ok(1, s1, P) and self.reaches(0, P[vertex(*d0)] + [0, 0, 0.03]) and self.reaches(1, P[vertex(*d1)] + [0, 0, 0.03]):
                lift = self.rng.uniform(0.7, 1.0)
                return {0: self.pick(0, s0, self.fold_path(d0, lift), speed), 1: self.pick(1, s1, self.fold_path(d1, lift), speed)}

    def one_fold(self, speed):
        P, r = self.cloth(), self.rng
        a = int(r.integers(2))
        for _ in range(20):
            i, j = EDGE[int(r.integers(len(EDGE)))]
            dst = [(i, N - 1 - j), (N - 1 - i, j), (N - 1 - i, N - 1 - j), (N // 2, N // 2)][int(r.integers(4))]
            if dst != (i, j) and self.ok(a, (i, j), P, own_side=True) and self.reaches(a, P[vertex(*dst)] + [0, 0, 0.03]):
                return {a: self.pick(a, (i, j), self.fold_path(dst, r.uniform(0.7, 1.0)), speed)}

    def bi_lift(self, speed):
        P, r = self.cloth(), self.rng
        for _ in range(12):
            src, _ = self.edge_pairs()
            s0, s1 = self.by_x(*src, P)
            if self.ok(0, s0, P) and self.ok(1, s1, P):
                together = (P[vertex(*s1)] - P[vertex(*s0)]) * np.array([1, 1, 0])
                offs, z = [], 0.0
                for k in range(int(r.integers(1, 4))):
                    z = r.uniform(0.06, 0.20)
                    offs.append((np.array([r.uniform(-0.05, 0.05), r.uniform(-0.12, 0.12), z]), r.uniform(0.0, 0.2)))  # shift, slack
                drop = 1.0 if r.random() < 0.6 else r.uniform(0, 0.6)
                return {a: self.pick(a, s, self.way_path([o + sign * slack * together / 2 for o, slack in offs], drop), speed) for a, s, sign in ((0, s0, 1), (1, s1, -1))}

    def carry(self, speed):
        P, r = self.cloth(), self.rng
        a = int(r.integers(2))
        for _ in range(20):
            ij = (int(r.choice(GRID)), int(r.choice(GRID)))
            if self.ok(a, ij, P, own_side=True):
                offs = [np.array([r.uniform(-0.12, 0.12), r.uniform(-0.12, 0.12), r.uniform(0.06, 0.20)]) for _ in range(int(r.integers(1, 4)))]
                return {a: self.pick(a, ij, self.way_path(offs, r.uniform(0, 1)), speed)}

    def drag(self, speed):
        P, r = self.cloth(), self.rng
        a = int(r.integers(2))
        for _ in range(20):
            ij = EDGE[int(r.integers(len(EDGE)))]
            if self.ok(a, ij, P, own_side=True):
                shift = np.r_[-P[:, :2].mean(0) + r.uniform(-0.07, 0.07, 2), r.uniform(0.015, 0.04)]  # mostly back towards the middle
                return {a: self.pick(a, ij, self.way_path([shift], 0.0), speed)}

    def __iter__(self):
        r = self.rng
        moves, probs = [self.bi_fold, self.one_fold, self.bi_lift, self.carry, self.drag], [0.27, 0.23, 0.17, 0.18, 0.15]
        while True:
            for _ in range(int(r.integers(1, 7))):
                yield np.stack(self.cur), np.zeros(2), []
            plan = moves[int(r.choice(5, p=probs))](float(np.exp(r.uniform(np.log(0.6), np.log(1.7)))))
            if plan is None:
                continue
            closed = np.zeros(2)
            while plan:
                events = []
                for a in list(plan):
                    step = next(plan[a], None)
                    if step is None:
                        del plan[a]
                        closed[a] = 0.0
                        continue
                    self.cur[a], closed[a] = np.asarray(step[0], float), step[1]
                    if step[2]:
                        events.append(step[2])
                if plan:
                    yield np.stack(self.cur), closed.copy(), events


class GraspPlan:
    """One reach, grasp, carry, release per arm; the reach follows the chosen vertex wherever the cloth is at that moment."""

    def __init__(self, rng, arm_index, vertices, vertex_pos, frames):
        side = -1 if arm_index == 0 else 1
        self.home = np.array([side * 0.24, rng.uniform(-0.05, 0.05), 0.17])
        self.active = rng.random() < 0.85
        near = [v for v in vertices if side * vertex_pos[v][0] > -0.03] or list(vertices)
        self.v = int(rng.choice(near))
        t = int(rng.integers(3, 25))
        self.t_above, self.t_down, self.t_close = t + 22, t + 36, t + 40
        self.t_carry, self.t_lower, self.t_open, self.t_back = self.t_close + 32, self.t_close + 46, self.t_close + 50, self.t_close + 70
        self.t0 = t
        self.move = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(0.08, 0.20)])
        self.drop = rng.uniform(0.0, 1.0)  # how far it comes back down before letting go
        self.grab = self.carry = self.release = None

    def step(self, t, vpos, pinch):
        """-> commanded position, gripper closed, event ('attach' | 'detach' | None)."""
        if not self.active or t < self.t0:
            return self.home, 0.0, None
        u = lambda a, b: 0.5 - 0.5 * np.cos(np.pi * np.clip((t - a) / max(b - a, 1), 0, 1))
        above = vpos + np.array([0, 0, 0.09])
        if t < self.t_above:
            return self.home + (above - self.home) * u(self.t0, self.t_above), 0.0, None
        if t < self.t_down:
            return above + (np.array([vpos[0], vpos[1], max(vpos[2], TABLE_Z)]) - above) * u(self.t_above, self.t_down), 0.0, None
        if t < self.t_close:
            if self.grab is None:
                self.grab = np.array([vpos[0], vpos[1], max(vpos[2], TABLE_Z)])
            return self.grab, 1.0, "attach" if t == self.t_close - 1 else None
        if self.carry is None:
            self.carry = np.clip(self.grab + self.move, [-0.22, -0.22, 0.05], [0.22, 0.22, 0.26])
            self.release = self.carry.copy()
            self.release[2] = self.carry[2] + (TABLE_Z + 0.02 - self.carry[2]) * self.drop
        if t < self.t_carry:
            return self.grab + (self.carry - self.grab) * u(self.t_close, self.t_carry), 1.0, None
        if t < self.t_lower:
            return self.carry + (self.release - self.carry) * u(self.t_carry, self.t_lower), 1.0, None
        if t < self.t_open:
            return self.release, 0.0, "detach" if t == self.t_lower else None
        up = self.release + np.array([0, 0, 0.08])
        return self.release + (up - self.release) * u(self.t_open, self.t_back), 0.0, None


def simulate(mode, seed, frames):
    """-> build parameters, qpos (frames, nq), action (frames, 14), pinch (frames, 2, 3), cloth extents (2, 3)."""
    rng = np.random.default_rng(seed)
    pose = dict(x=0.0, y=0.0, yaw=0.0) if mode == "hold" else dict(x=float(rng.uniform(-0.04, 0.04)), y=float(rng.uniform(-0.04, 0.04)), yaw=float(rng.uniform(-0.4, 0.4)))
    m, arms = build(mode, **pose)
    d, dk = mujoco.MjData(m), mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    vert = lambda v: d.xpos[m.body(f"cloth_{v}").id].copy()
    home_q = np.array(ROBOT["home"], float)
    cloth_ids = np.array([m.body(f"cloth_{v}").id for v in range(N * N)])
    script = None
    if mode == "hold":
        start = np.stack([vert(next(iter(arm["eq"]))) for arm in arms])
        start[:, 2] = TABLE_Z
        targets, closed = plan_hold(rng, start, frames)
        first = start
    elif mode == "fold":
        script = Script(rng, lambda: d.xpos[cloth_ids].copy(), [arm["base"] for arm in arms])
        first = np.stack(script.home)
        steps = iter(script)
    else:
        plans = [GraspPlan(rng, a, list(arms[a]["eq"]), {v: vert(v) for v in arms[a]["eq"]}, frames) for a in range(2)]
        if not any(p.active for p in plans):
            plans[int(rng.integers(2))].active = True
        first = np.stack([p.home for p in plans])
    q = [solve_ik(m, dk, arm, first[a], home_q, iters=200) for a, arm in enumerate(arms)]
    for a, arm in enumerate(arms):
        d.qpos[arm["qadr"]] = q[a]
        grip = CLOSED if mode == "hold" else OPEN
        d.qpos[arm["fingers"]] = [grip, -grip]
        d.ctrl[arm["act"]], d.ctrl[arm["grip"]] = q[a], grip
        if mode == "hold":
            d.eq_active[next(iter(arm["eq"].values()))] = 1
    for _ in range(250):  # settle
        mujoco.mj_step(m, d)
    sub = int(round(1 / FPS / DT))
    qpos, action, pinch = [], np.zeros((frames, 14)), []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for t in range(frames):
        prev = d.ctrl.copy()
        goal = prev.copy()
        if script is not None:
            s_tgt, s_closed, s_events = next(steps)
            for ev in s_events:
                if ev[1] == "attach":
                    d.eq_active[arms[ev[0]]["eq"][ev[2]]] = 1
                else:
                    d.eq_active[list(arms[ev[0]]["eq"].values())] = 0
        for a, arm in enumerate(arms):
            if script is not None:
                tgt, cl = s_tgt[a], s_closed[a]
            elif mode == "hold":
                tgt, cl, event = targets[t, a], closed[t, a], None
            else:
                tgt, cl, event = plans[a].step(t, vert(plans[a].v), held_point(d, arm))
                if event == "attach":
                    d.eq_active[arm["eq"][plans[a].v]] = 1
                elif event == "detach":
                    d.eq_active[list(arm["eq"].values())] = 0
            q[a] = solve_ik(m, dk, arm, np.asarray(tgt, float), q[a])
            goal[arm["act"]], goal[arm["grip"]] = q[a], CLOSED if cl else OPEN
            action[t, a * 7 : a * 7 + 3], action[t, a * 7 + 6] = tgt, cl
        for s in range(sub):
            d.ctrl[:] = prev + (goal - prev) * (s + 1) / sub
            mujoco.mj_step(m, d)
        qpos.append(d.qpos.copy())
        pinch.append(np.stack([held_point(d, arm) for arm in arms]))
        pts = np.concatenate([d.flexvert_xpos, pinch[-1]])
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    assert np.isfinite(np.stack(qpos)).all(), f"episode {seed} diverged"
    return pose, np.stack(qpos), action, np.stack(pinch), np.stack([lo, hi])


def sim_job(job):
    mode, ep, out, frames = job
    path = Path(out) / "states" / f"episode_{ep}.npz"
    if not path.exists():
        pose, qpos, action, pinch, ext = simulate(mode, 1000 + ep, frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path) + ".tmp.npz", qpos=qpos, action=action, pinch=pinch, ext=ext, **pose)
        os.replace(str(path) + ".tmp.npz", path)
    return np.load(path)["ext"]


def fit_grid(ext, cells=64, margin=0.04):
    """Workspace box around everything the cloth and the grippers reach, and a grid of cubic cells with ``cells`` along its
    longest side (every side a multiple of 8 cells). -> bbox (6,), grid (3,)"""
    lo, hi = ext[:, 0].min(0) - margin, ext[:, 1].max(0) + margin
    lo[2] = -0.02
    v = (hi - lo).max() / cells
    grid = (np.ceil((hi - lo) / v / 8) * 8).astype(int)
    centre = (lo + hi) / 2
    centre[2] = lo[2] + grid[2] * v / 2  # the box grows upwards only
    lo, hi = centre - grid * v / 2, centre + grid * v / 2
    return [round(float(x), 4) for x in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])], [int(g) for g in grid]


def packbits(mask):
    return np.packbits(mask.reshape(mask.shape[0], -1), axis=1)


def render_job(job):
    mode, ep, out, bbox = job
    path = Path(out) / "packed" / f"episode_{ep}.hdf5"
    if path.exists():
        return ep, "exists"
    s = np.load(Path(out) / "states" / f"episode_{ep}.npz")
    m, arms = build(mode, float(s["x"]), float(s["y"]), float(s["yaw"]))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    frames, nc = len(s["qpos"]), len(CAMS)
    rgb, depth, obj = np.zeros((nc, frames, H, W, 3), np.uint8), np.zeros((nc, frames, H, W), np.float16), np.zeros((nc, frames, H, W), np.uint8)
    for t in range(frames):
        d.qpos[:] = s["qpos"][t]
        mujoco.mj_forward(m, d)
        for c in range(nc):
            r.update_scene(d, camera=c)
            rgb[c, t] = r.render()
            r.enable_depth_rendering()
            z = r.render()
            r.disable_depth_rendering()
            z[z > 5.0] = 0.0
            depth[c, t] = z
            r.enable_segmentation_rendering()
            seg = r.render()
            r.disable_segmentation_rendering()
            obj[c, t] = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_FLEX)
    f_ = 0.5 * H / np.tan(np.radians(FOVY) / 2)
    K = np.array([[f_, 0, W / 2], [0, f_, H / 2], [0, 0, 1]], np.float32)
    b = bbox
    jpg = lambda im: np.frombuffer(cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8)
    vlen = h5py.vlen_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs.update(episode=ep, num_frames=frames, H=H, W=W, cam_ids=np.arange(nc, dtype=np.int32), bbox=np.array(b, np.float32), has_obj_mask=True, dataset=Path(out).name, format_version="packed_v3", depth_semantics="full_scene_lrc", obj_labels=json.dumps({"1": "cloth"}), source=f"mujoco flex cloth, two {os.environ.get('SIM_ARM', 'vx300s')} arms, {mode}")
        f.create_dataset("action/cartesian_action", data=s["action"])
        f.create_dataset("sim/pinch", data=s["pinch"].astype(np.float32))
        f.create_dataset("sim/qpos", data=s["qpos"].astype(np.float32), compression="gzip")
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        for c in range(nc):
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera -> x right, y down, z forward
            E[:3, 3] = d.cam_xpos[c]
            z = depth[c].astype(np.float32)
            pw = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], -1) @ E[:3, :3].T + E[:3, 3]
            valid = (z > 0.05) & (pw[..., 0] > b[0]) & (pw[..., 0] < b[1]) & (pw[..., 1] > b[2]) & (pw[..., 1] < b[3]) & (pw[..., 2] > b[4]) & (pw[..., 2] < b[5])
            g = f.create_group(f"img/camera_{c}")
            g.create_dataset("depth", data=depth[c], compression="gzip", compression_opts=4, shuffle=True, chunks=(1, H, W))
            g.create_dataset("valid_mask", data=packbits(valid), compression="gzip", chunks=(1, (H * W + 7) // 8))
            g.create_dataset("obj_mask", data=obj[c], compression="gzip", chunks=(1, H, W))
            ds = g.create_dataset("rgb_in_ir", (frames,), dtype=vlen)
            rd = f.create_group(f"render/camera_{c}")
            rd.attrs.update(H=H, W=W, crop=np.array([0, 0, W, H]))
            rj = rd.create_dataset("jpg", (frames,), dtype=vlen)
            for t in range(frames):
                ds[t] = rj[t] = jpg(rgb[c, t])
            f.create_dataset(f"cam/camera_{c}/K_ir", data=np.repeat(K[None], frames, 0))
            f.create_dataset(f"cam/camera_{c}/E_cam2world", data=np.repeat(E[None], frames, 0))
            rd.create_dataset("K", data=np.repeat(K[None], frames, 0))
            rd.create_dataset("viewmat", data=np.repeat(np.linalg.inv(E)[None], frames, 0).astype(np.float32))
            rd.create_dataset("obj_mask", data=packbits(obj[c] > 0), compression="gzip", chunks=(1, (H * W + 7) // 8))
    os.replace(tmp, path)
    return ep, f"{frames} frames"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("hold", "grasp", "fold"))
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--val", type=int, default=10, help="the last this-many episodes are the validation set")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--render-workers", type=int, default=None, help="rendering holds an episode of images in memory (default: --workers)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        ext = np.stack(list(ex.map(sim_job, [(a.mode, e, a.out, a.frames) for e in range(a.episodes)])))
        bbox, grid = fit_grid(ext)
        json.dump(dict(bbox=bbox, grid=grid, x_range=bbox[0:2], y_range=bbox[2:4], z_range=bbox[4:6]), open(out / "grid.json", "w"), indent=1)
        print(f"simulated {a.episodes} episodes; workspace {bbox}, grid {grid}, cell {1000 * (bbox[1] - bbox[0]) / grid[0]:.1f} mm", flush=True)
    with ProcessPoolExecutor(a.render_workers or a.workers) as ex:
        for ep, msg in ex.map(render_job, [(a.mode, e, a.out, bbox) for e in range(a.episodes)]):
            print(f"episode {ep}: {msg}", flush=True)
    eps = list(range(a.episodes))
    split = dict(train=eps[: -a.val], val=eps[-a.val :])
    json.dump(split, open(out / "splits.json", "w"))
    for name, ids in split.items():
        (out / name).mkdir(exist_ok=True)
        for e in ids:
            if not (out / name / f"episode_{e}.hdf5").exists():
                os.symlink(f"../packed/episode_{e}.hdf5", out / name / f"episode_{e}.hdf5")


if __name__ == "__main__":
    main()

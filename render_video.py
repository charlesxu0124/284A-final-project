"""Render trajectories saved by benchmark.py --output as a video, one panel per trajectory."""

import argparse
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, nargs="+", help="NPZ files with positions and triangles")
    parser.add_argument("--labels", nargs="+", help="panel titles, defaults to the file names")
    parser.add_argument("--environments", type=int, nargs="+", default=[0],
                        help="batch indices to draw; several indices of one file make one panel each")
    parser.add_argument("--fps", type=int, default=30, help="30 plays the 60 Hz rollout at half speed")
    parser.add_argument("--output", type=Path, required=True, help="MP4 file, needs ffmpeg on the path")
    args = parser.parse_args()

    labels = args.labels or [path.parent.name + "/" + path.stem for path in args.trajectories]
    if len(labels) != len(args.trajectories):
        parser.error("give one label per trajectory")
    panels = []
    for label, path in zip(labels, args.trajectories):
        with np.load(path) as data:
            for environment in args.environments:
                title = label if len(args.environments) == 1 else f"{label}, fold {environment}"
                panels.append((title, data["positions"][:, environment], data["triangles"], float(data["dt"])))
    frames = min(len(positions) for _, positions, _, _ in panels)
    side = max(positions[0, :, :2].max() for _, positions, _, _ in panels)
    height = max(positions[..., 2].max() for _, positions, _, _ in panels)

    columns = min(len(panels), 4)
    rows = -(-len(panels) // columns)
    figure = plt.figure(figsize=(4.2 * columns, 3.6 * rows + 0.3), dpi=100)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = None
    for frame in range(frames):
        figure.clf()
        for index, (title, positions, triangles, dt) in enumerate(panels, start=1):
            axis = figure.add_subplot(rows, columns, index, projection="3d")
            x, y, z = positions[frame].T
            axis.plot_trisurf(x, y, z, triangles=triangles, color=(0.25, 0.65, 0.40),
                              edgecolor=(0, 0, 0, 0.25), linewidth=0.2)
            axis.set(xlim=(0, side), ylim=(0, side), zlim=(0, max(height, 0.05)), xticks=[], yticks=[], zticks=[])
            axis.set_box_aspect((1, 1, max(height, 0.05) / side))
            axis.view_init(25, -60)
            axis.set_title(title, fontsize=10)
        figure.suptitle(f"t = {frame * dt:.2f} s", fontsize=10)
        figure.subplots_adjust(0, 0, 1, 0.92, 0, 0.08)
        figure.canvas.draw()
        image = np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[..., :3])
        if encoder is None:
            size = f"{image.shape[1]}x{image.shape[0]}"
            encoder = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(args.fps),
                 "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(args.output)], stdin=subprocess.PIPE)
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()

"""Add a simulated dataset to the 3D World Simulator's task list (tasks.py), with the grid cloth_arms.py fitted to it.

    python mujoco_cloth/register_task.py data/simcloth_fold_yam
"""
import json
import os
import sys
from pathlib import Path

WS_ROOT = os.environ.get("WS_ROOT")  # a clone of https://github.com/jwang771/3d-world-simulator
if not WS_ROOT:
    raise SystemExit("set WS_ROOT to your clone of the 3D World Simulator repository")
dataset = Path(sys.argv[1])
name = dataset.name
grid = json.load(open(dataset / "grid.json"))
tasks = Path(WS_ROOT) / "tasks.py"
text = tasks.read_text()
if f'"{name}"' in text:
    raise SystemExit(f"{name} is already in {tasks}")
entry = f'''    "{name}": dict(  # simulated cloth (mujoco_cloth/cloth_arms.py); grid from {name}/grid.json
        datasets=["{name}"],
        grid={tuple(grid["grid"])},
        x_range={tuple(grid["x_range"])},
        y_range={tuple(grid["y_range"])},
        z_range={tuple(grid["z_range"])},
        wrist_cams=[],
        tactile=False,
    ),
'''
head = "TASKS = {\n"
if head not in text:
    raise SystemExit(f"could not find the task list in {tasks}")
tasks.write_text(text.replace(head, head + entry, 1))
print(f"added {name}: grid {grid['grid']}, workspace {grid['bbox']}")

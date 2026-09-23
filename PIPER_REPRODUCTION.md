# Code as Policies on Piper/MuJoCo

## Purpose and boundary

This project preserves the released Google Code as Policies notebooks and prompts, then migrates the tabletop tasks to the `robot_control` Piper/MuJoCo backend. It reproduces the task semantics and hierarchical LMP procedure; it is not an identical rerun of the paper's UR5e, PyBullet, or hardware experiments.

The unchanged upstream files are in `original/`. `original/README.original.md` preserves the upstream README, and `original/notebooks/Interactive_Demo.ipynb` remains the source of the upstream tabletop LMP.

## Backend and environment

The backend is tracked as the `robot_control/` Git submodule and frozen per experiment in `manifest.json`. The adapter imports only `robot_control`, its MuJoCo backend, public API types, and sensor interfaces. The old compatibility package is not part of the execution path. The Piper adapter and the FR3 direct-control path have been checked against backend commit `47b9a8b345fba50b042e577845b3d3b02fe0893b`.

```bash
git submodule update --init --recursive
cd robot_control
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[mujoco,dev]' openai astunparse shapely
cd ..
export ROBOT_CONTROL_ROOT="$PWD/robot_control"
export PYTHONPATH="$ROBOT_CONTROL_ROOT/src:$PYTHONPATH"
```

The model credential is read from `OPENAI_API_KEY`. An optional OpenAI-compatible endpoint can be supplied through `OPENAI_BASE_URL`; neither value is stored in the repository or experiment artifacts.

## Control protocol

The tabletop is generated from fixed seeds by `scene_factory.py`. The registered protocol uses a forward offset of zero, an XY workspace of approximately `[-0.19, 0.01] × [-0.19, 0.19]`, 8 mm placement jitter, and a 15 cm minimum initial separation. The Piper tool orientation is fixed downward for this migration: `(0, 0, 1, 0)` in wxyz form.

`--require-downward-ik` checks every sampled object at the registered grasp, release, and transit heights. Failed candidates advance deterministically to the next seed and record both the accepted seed and the number of attempts. IK success does not guarantee contact, collision-free motion, or placement success.

`workspace_calibration.py` probes the same backend over a 1 cm grid and writes a local calibration JSON. The calibration hash and backend revision are included in the canonical manifest. A run refuses to start if its calibration belongs to another backend commit, even when the change does not affect Piper. Recompute calibration after every backend update:

```bash
"$ROBOT_CONTROL_ROOT/.venv/bin/mjpython" workspace_calibration.py \
  --output results/canonical/workspace-calibration.json
```

## RGB-D interface

Simulation and hardware-facing code share the `RGBDPerception` contract. A camera provides RGB, depth, intrinsics, and an extrinsic transform relative to the robot base. Segmentation masks are converted to 3D points by depth back-projection, then transformed into the robot/world frame. Depth values that are zero, non-finite, outside the configured range, or robust statistical outliers within a mask are excluded before position, box, and color estimation. If no valid inlier remains, the object is omitted from the observation set.

```python
from robot_control.sensors.mujoco_rgbd import MujocoRGBDCamera
from rgbd_perception import RGBDPerception, mujoco_world_from_camera

camera = MujocoRGBDCamera(tabletop.backend.model, tabletop.backend.data,
                          width=640, height=480)
camera.connect()
perception = RGBDPerception(
    camera, segmenter,
    lambda frame: mujoco_world_from_camera(tabletop.backend, frame),
)
observations = perception.refresh()
```

The optional segmentation service is configured through runtime environment variables and is not part of the public experiment record.

The bundled `sam3_client.py` does not use HTTP. It opens a TCP socket to `SAM3_HOST:SAM3_PORT` and exchanges little-endian length-prefixed JSON and binary array frames. The default port is `28317`; the host and port are runtime configuration only.

## Artifacts

Each canonical run contains `manifest.json`, `trials.jsonl`, `summary.json`, deterministic scene XML files, and optional representative videos. The manifest records code, prompt, scene, package, and backend revisions. The summary reports category rates, Wilson 95% intervals, and failure stages.

## Real Franka path

`franka_real.py` and `run_franka_real.py` provide a separate FR3 hardware path. They reuse the archived hierarchical LMP and paper prompts, but do not reuse MuJoCo object-state queries. The operator supplies object observations in the FR3 base frame, the adapter plans Cartesian waypoints with the measured end-effector orientation, and hardware motion is disabled unless `--execute --confirm-real` are both present. The latest backend exposes direct FCI control through `pylibfranka`; actual hardware validation remains the operator's responsibility.

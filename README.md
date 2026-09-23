# Code as Policies: Piper/MuJoCo Reproduction

This repository preserves the released Google `code_as_policies` implementation and adds a reproducible Piper/MuJoCo migration of its tabletop tasks, prompts, hierarchical LMP execution, and scoring protocol.

## Scope

- The upstream README is preserved at `original/README.original.md`.
- All upstream notebooks are preserved unchanged in `original/notebooks/`.
- The policy-generation path still loads the upstream `Interactive_Demo.ipynb` directly.
- The simulator backend is the pinned `robot_control/` Git submodule, frozen per run in `manifest.json`.
- This is a Piper/MuJoCo migration study, not a claim of identical UR5e/PyBullet hardware results.

## Reproduction flow

```text
language instruction
        ↓
upstream tabletop few-shot prompts
        ↓
tabletop_ui LMP generates Python policy
        ↓
recursive generation of undefined helper functions
        ↓
MuJoCo state queries or RGB-D object observations
        ↓
Piper pick/place primitives
        ↓
deterministic scoring, failure classification, and artifacts
```

`piper_lmp.py` executes the upstream `LMP`, `LMPFGen`, `FunctionParser`, prompts, and configuration. The adapter changes only the robot, scene, camera, and execution interfaces.

## Experiments

The paper suite contains 22 task categories: 8 seen/seen, 8 seen/unseen, and 6 unseen/unseen categories. The canonical protocol runs 50 trials per category, for 1100 trials total, with fixed seeds, deterministic scene sampling, downward-tool IK preflight, per-trial XML scenes, and complete JSONL records.

The main track uses structured MuJoCo object state queries to stay closest to the paper simulation protocol. The optional perception track uses the same task cases with a wrist RGB-D camera and an external SAM3-compatible segmenter; its results are reported separately. The perception adapter rejects zero, non-finite, out-of-range, and robust depth outlier pixels before computing object observations. The bundled SAM3 client uses a length-prefixed TCP socket protocol, not HTTP.

The fixed tool orientation is downward for this Piper migration. This is an adapter constraint, not a claim that every original Code as Policies robot policy used only one orientation.

## Layout

```text
original/                         unchanged upstream README and notebooks
paper_prompts/                    released simulation prompts
scenes/                           Piper/MuJoCo scene templates
robot_control/                    pinned Piper/MuJoCo backend Git submodule
franka_real.py                    safety-gated FR3 real-robot tabletop adapter
run_franka_real.py                dry-run and explicit-execution entry point
piper_lmp.py                      upstream LMP loading and execution adapter
piper_demo.py                     Piper/MuJoCo control and RGB-D adapter
scene_factory.py                  deterministic scene generation
workspace_calibration.py          backend-specific IK workspace probe
paper_suite.py                    22 paper task categories
task_scoring.py                   registered Piper scoring protocol
rgbd_perception.py                RGB-D reprojection and observations
sam3_client.py                    generic TCP segmentation client
run_paper_suite.py                planning, execution, recording, and summaries
repro_metadata.py                 manifests, version capture, and confidence intervals
results/                          canonical results and local historical archives
tests/                            offline and MuJoCo tests
```

## Setup

Clone with submodules and install the pinned backend inside this checkout:

```bash
git clone --recurse-submodules https://github.com/HY-LiYihan/code_as_policy.git
cd code_as_policy
# If this repository was cloned without --recurse-submodules:
git submodule update --init --recursive
cd robot_control
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[mujoco,dev]' openai astunparse shapely
cd ..
export ROBOT_CONTROL_ROOT="$PWD/robot_control"
export PYTHONPATH="$ROBOT_CONTROL_ROOT/src:$PYTHONPATH"
```

Canonical runs require a clean backend worktree at the commit pinned by this repository. Do not update the submodule to a floating branch without explicitly recording the new Git link. The manifest stores the backend commit and nested submodules, package versions, prompt hashes, scene protocol, and adapter hashes. The API credential is read only from `OPENAI_API_KEY` at runtime.

## Validation and runs

```bash
"$ROBOT_CONTROL_ROOT/.venv/bin/mjpython" -m pytest -q tests
```

Plan scenes without model calls:

```bash
"$ROBOT_CONTROL_ROOT/.venv/bin/python" run_paper_suite.py \
  --plan-only --seed 20260923 --trials 50 --require-downward-ik \
  --output-dir results/canonical/paper-sim-plan
```

Run the canonical OpenAI policy track:

```bash
export OPENAI_API_KEY='set outside the repository'
export OPENAI_BASE_URL='https://api.openai.com/v1'
"$ROBOT_CONTROL_ROOT/.venv/bin/mjpython" run_paper_suite.py \
  --seed 20260923 --trials 50 --require-downward-ik \
  --record-representative-videos \
  --output-dir results/canonical/openai-paper-sim
```

The representative-video mode keeps the first successful and first failed/exceptional video per experiment group. All trials remain in `trials.jsonl`; `summary.json` contains category rates, Wilson 95% intervals, and failure-stage counts.

## Real Franka FR3

The real-robot path reuses the original hierarchical LMP but is separate from the Piper/MuJoCo experiment runner. It accepts object observations already expressed in the FR3 base frame, so camera calibration is explicit and auditable. The default is a dry run; physical motion requires both `--execute` and `--confirm-real`.

```bash
export OPENAI_API_KEY='set outside the repository'
export OPENAI_BASE_URL='https://api.openai.com/v1'
PYTHONPATH="$PWD/robot_control/src" \
  "$ROBOT_CONTROL_ROOT/.venv/bin/python" run_franka_real.py \
  --objects-json examples/franka_objects.base-frame.json \
  --command "Pick up the blue block and place it on the yellow bowl"
```

On the Ubuntu control PC, install the matching `pylibfranka`/`libfranka` pair, set `FRANKA_ROBOT_IP`, and inspect the generated plan before enabling motion. Do not use the execution flags until the robot workspace is clear and the emergency stop is reachable:

```bash
PYTHONPATH="$PWD/robot_control/src" \
  "$ROBOT_CONTROL_ROOT/.venv/bin/python" run_franka_real.py \
  --objects-json examples/franka_objects.base-frame.json \
  --command "Pick up the blue block and place it on the yellow bowl" \
  --robot-ip "$FRANKA_ROBOT_IP" --execute --confirm-real
```

The current backend provides standalone RealSense frames for FR3 but does not provide calibrated FR3 camera extrinsics. Therefore the JSON input must come from a separately calibrated perception process; this repository does not claim automatic FR3 RGB-D-to-base calibration.

## Results and privacy

`results/canonical/` contains the public experiment artifacts selected for release. Local historical and intermediate artifacts remain under `results/archive/` but are excluded from the public source push. No credentials, service addresses, or machine-specific absolute paths belong in the repository.

The released result must be described as a Code as Policies Piper/MuJoCo migration under the registered adapter protocol. It must not be presented as the original paper's hardware success rate.

# Paper Simulation Alignment Protocol

## Reproduction target

The experiment reproduces the public Code as Policies tabletop task categories, simulation prompts, and hierarchical LMP generation procedure. The upstream implementation is loaded from `original/notebooks/Interactive_Demo.ipynb`; Piper/MuJoCo control and perception are migration adapters.

The backend revision, submodules, package versions, prompt hashes, scene protocol, and adapter hashes are recorded in every `manifest.json`.

## Task groups

| Group | Categories | Count |
| --- | --- | ---: |
| seen instructions / seen attributes | `seen_1`–`seen_8` | 8 |
| seen instructions / unseen attributes | `seen_1`–`seen_8` | 8 |
| unseen instructions / unseen attributes | `unseen_1`–`unseen_6` | 6 |

The canonical protocol runs 50 trials per category, for 1100 trials total. Trial seeds, accepted scene seeds, sampling attempts, and XML scenes are retained.

## Alignment and differences

Aligned with the released procedure:

1. tabletop task semantics and public simulation prompts;
2. the `tabletop_ui` high-level LMP;
3. recursive helper generation for object names, positions, questions, and shape transforms;
4. generated Python policies calling robot primitives;
5. seen/unseen instruction and attribute groups;
6. per-trial generated code, actions, exceptions, and final states.

Migration-specific changes:

- UR5e/PyBullet is replaced by Piper/MuJoCo;
- the original vision interface is optionally replaced by wrist RGB-D observations;
- scene seeds and robot reachability are registered by this repository;
- geometric success thresholds are registered Piper scoring rules, not published paper thresholds.

The result is therefore a Piper/MuJoCo migration reproduction and must not be reported as the original hardware success rate. The separate FR3 hardware entry point is an adapter validation path, not part of the 22-category simulation statistics.

## Scene and reachability protocol

The scene factory uses six deterministic slots, up to 8 mm jitter, a 15 cm minimum separation, and the fixed downward tool orientation. Candidate scenes are checked at heights `0.92`, `0.93`, `0.965`, and `0.97` m. A failed candidate advances to `case.seed + attempt`, with a maximum of 128 attempts.

The separate workspace calibration uses the frozen backend and a 1 cm grid. It documents the safe sampling region but does not replace per-trial preflight.

## Running the suite

```bash
"$ROBOT_CONTROL_ROOT/.venv/bin/python" run_paper_suite.py \
  --plan-only --seed 20260923 --trials 50 --require-downward-ik \
  --output-dir results/canonical/paper-sim-plan
```

```bash
export OPENAI_API_KEY='set outside the repository'
export OPENAI_BASE_URL='https://api.openai.com/v1'
"$ROBOT_CONTROL_ROOT/.venv/bin/mjpython" run_paper_suite.py \
  --seed 20260923 --trials 50 --require-downward-ik \
  --record-representative-videos \
  --output-dir results/canonical/openai-paper-sim
```

The optional RGB-D track uses the same cases and seeds but writes to a separate directory. Main-track and perception-track rates must never be merged.

## Scoring and failures

Scoring evaluates action geometry before and after execution: relative directions, distances, bowl placement, stacking, corners, sides, and lines. These thresholds are Piper migration rules.

Failures are classified as `generation`, `ik`, `execution`, `grasp`, `placement`, `scoring`, or `setup`. `summary.json` reports rates, Wilson 95% intervals, and counts by failure stage. Planned-only records have no success rate.

## Release hygiene

Public files contain no credentials, machine-specific absolute paths, or local environment details. Historical artifacts remain available locally for audit but are excluded from the public source push.

# Experiment Results

## Canonical results

`canonical/` contains release-ready experiment manifests, summaries, and trial records. Each run records the frozen backend revision, prompt hashes, scene protocol, package versions, and implementation hashes.

The main public result is the OpenAI policy track. Perception-enabled runs use separate directories and must not be combined with the main-track rate.

## Local archives

`archive/` contains historical and intermediate artifacts retained locally for audit. These files may reflect earlier scene layouts or incomplete planned-only runs and are excluded from the public source push.

The interrupted September 24 main-track run (309/1100 trials) is stored locally in `archive/intermediate/openai-paper-sim-20260924-partial-309/`. It has no final summary and must not be presented as a completed canonical result. Its manifest records the backend used during execution; a calibration from a different backend revision was referenced, so its calibration claim is not valid for that run.

Generated videos, images, scene XML files, and local archives are intentionally ignored by Git. The public source contains structured experiment results and the documentation needed to reproduce them without credentials or machine-specific paths.

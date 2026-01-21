# MGC Pipeline Contract

This document defines what each pipeline step MUST produce and what it MUST NOT do.
If behavior changes, update this doc and the corresponding contract checks.

## Global rules

- Determinism: when `--deterministic` is set, outputs must be byte-stable across runs.
- Idempotency: re-running a step with the same inputs should not corrupt or duplicate outputs.
- No hidden mutation: steps should not write outside their declared output directories.
- Exit codes: if a step cannot produce its declared outputs, it MUST exit non-zero.

## Directories and key files

Common outputs (paths are examples; actual paths come from CLI args/env):
- Drop bundle directory: `<out_dir>/drop_bundle/`
- Evidence directory: `<out_dir>/` or `MGC_EVIDENCE_DIR`
- Submission zip: `<out_path>/submission.zip`
- Web build output: `<web_out_dir>/`

## Step: run daily

Command:
- `python -m mgc.main run daily --context <ctx> --out-dir <dir> [--seed N] [--deterministic]`

Produces (MUST):
- `<out_dir>/playlist.json`
- `<out_dir>/drop_bundle/daily_evidence.json` (or `<out_dir>/drop_evidence.json` if that’s your current canonical path)
- Any generated audio referenced by playlist (either in repo data dir or copied into bundle, depending on design)

Must NOT:
- Fail silently while returning exit 0.
- Change outputs when `--deterministic` and inputs are unchanged.

## Step: run publish-marketing

Command:
- `python -m mgc.main run publish-marketing --out-dir <dir> [--deterministic]`

Produces (MUST):
- Publish receipts (file-based audit trail) under a stable directory
- If it produces “post plans” or “assets”, those paths must be deterministic

Must NOT:
- Post to real networks by default (unless an explicit flag/credential is present)
- Create non-deterministic timestamps in receipts in deterministic mode

## Step: run drop

Command:
- `python -m mgc.main run drop --context <ctx> --out-dir <dir> [--seed N] [--deterministic]`

Produces (MUST):
- A complete “drop bundle directory” containing:
  - `playlist.json`
  - `daily_evidence.json` (or canonical evidence file)
  - Required audio payloads or resolvable references (depending on your design)

## Step: submission build

Command (canonical):
- `python -m mgc.main submission build --bundle-dir <drop_bundle_dir> --out <submission.zip>`

Produces (MUST):
- `<out>` exists and is non-empty
- Zip contents are deterministic:
  - stable ordering
  - stable timestamps
  - stable README content (if present)

Must NOT:
- Exit 0 without writing `<out>`

## Step: web build

Command:
- `python -m mgc.main web build --playlist <playlist.json> --out-dir <web_dir> [--clean] [--fail-if-none-copied]`

Produces (MUST):
- `<web_dir>/web_manifest.json`
- copied audio assets (unless explicitly configured otherwise)

Must NOT:
- Succeed if referenced tracks are missing when `--fail-on-missing` is set

## Step: run weekly

Command:
- `python -m mgc.main run weekly --context <ctx> --out-dir <dir> [--seed N] [--deterministic]`

Produces (MUST):
- `<out_dir>/playlist.json` (weekly playlist contract)
- Optional: a manifest describing what period it represents

Must NOT:
- Generate new audio (weekly should assemble / select, not synthesize)

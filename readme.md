# Music Generator Company (MGC)

## Overview

**Music Generator Company (MGC)** is a deterministic, autonomous release pipeline that demonstrates how generated media can be safely built, validated, approved, and published using modern CI/CD and release-engineering practices.

The project is intentionally designed with clear contracts, reproducible artifacts, explicit approval gates, and auditable outputs — without relying on paid APIs or external infrastructure.

**Live Demo (GitHub Pages):**  
https://skitsnygg.github.io/music-generator-company/

---

## System Goals

- End-to-end automation from **generation → release bundle → publish**
- Deterministic, CI-verifiable outputs
- Explicit separation between **build** and **publish**
- Approval-gated releases to prevent accidental shipping
- Fully auditable artifacts, receipts, and manifests
- No required paid APIs or cloud services

---

## Autonomous Release Pipeline

This project demonstrates a **full release lifecycle**, not just task automation.

### High-Level Flow

mgc run autonomous  
↓  
release bundle (audio + metadata)  
↓  
release contract (local / publish)  
↓  
CI determinism verification  
↓  
APPROVED (manual gate)  
↓  
mgc run publish  
↓  
GitHub Pages deploy

### What This Proves

- Deterministic builds suitable for CI
- Explicit contracts defining what “done” means
- Safe automation that cannot publish without approval
- Clear artifact boundaries between build and ship
- Real-world release discipline applied to generated media

---

## Key Concepts

### Release Bundles

A release bundle is a **self-contained directory** produced by `mgc run autonomous`.

It includes:
- Generated audio
- Playlist JSON
- Release manifest
- Evidence and receipts
- Optional static web player

Everything needed to inspect, verify, or publish the release lives in one folder.

---

### Release Contracts

Release contracts define **required artifacts** and **constraints**.

Supported modes:
- `local` — build-only validation
- `publish` — requires marketing receipts and web bundle

Contracts are validated automatically and written to `contract_report.json`.  
CI fails if a contract fails.

---

### Determinism

Determinism is enforced across:
- Audio generation (stub provider)
- JSON manifests
- Web bundles
- Submission archives

CI verifies identical inputs produce identical outputs across full directory trees.

---

### Approval Gating

Publishing is blocked unless explicitly approved.

Approval mechanism:

touch <bundle-dir>/APPROVED

Without this file, publishing is refused.

---

### Publish vs Build (Strict Separation)

- `run autonomous` builds a release
- `run publish` ships an already-built release

Publishing never rebuilds or mutates artifacts.

---

## Architecture

### Core Components

**CLI / Orchestration**
- Single entrypoint (`mgc`)
- Explicit lifecycle subcommands
- JSON-first outputs

**Music Generation**
- Stub provider (default, deterministic)
- Optional Riffusion provider (local service)
- Suno (scaffolded) + DiffSinger provider (HTTP or local command wrapper)

**Playlist Builder**
- Deterministic selection and ordering
- Stable playlist JSON
- Optional provider filter via `MGC_PLAYLIST_PROVIDER` (e.g. riffusion)

**Marketing System**
- Plans posts and records receipts
- No live APIs required
- Receipts staged and auditable
- File-based publish flow (marketing/publish -> marketing/receipts)

**Web Player**
- Static HTML + JS
- Built from playlist JSON
- Deployed via GitHub Pages

---

## Release Bundle Layout

out_dir/  
├─ contract_report.json  
├─ drop_evidence.json  
├─ manifest.json  
├─ playlist.json  
├─ tracks/<track_id>.wav  
├─ marketing/receipts/receipts.jsonl  
└─ web/  
   ├─ index.html  
   ├─ playlist.json  
   ├─ web_manifest.json  
   └─ tracks/<track_id>.wav  

The bundle is portable, auditable, deployable, and deterministic.

---

## How to Run

### Local Setup

python -m venv .venv  
source .venv/bin/activate  
pip install -U pip  
pip install -e .

No API keys are required for the default stub provider.

### Docker (Optional)

Build and run the offline end-to-end verification inside a container:

docker build -t mgc .  
docker run --rm mgc

Or with compose:

docker compose run --rm mgc

If Docker Hub is slow/unreachable, use a public mirror:

docker build -t mgc --build-arg PY_BASE=public.ecr.aws/docker/library/python:3.12-slim .  
docker run --rm mgc

### End-to-end verification (offline-friendly)

Run a full deterministic smoke that exercises music generation (stub), playlists, drop, web build,
submission zip, marketing receipts, and billing CLI using the fixture DB:

scripts/verify_all.sh

### Marketing Publish (Receipts)

Daily/weekly runs emit deterministic post payloads under `marketing/publish/`.  
Publishing is file-based and writes receipts to `marketing/receipts/`.
If `ffmpeg` is available, a short MP4 teaser is generated under `marketing/media/`.

Environment toggles:
- `MGC_PUBLISH_MARKETING=1` (default) — run `publish-marketing` in `scripts/run_daily.sh`
- `MGC_MARKETING_PLATFORMS="x,tiktok,instagram_reels,youtube_shorts"` — platforms for publish payloads

Manual publish (file mode):
python -m mgc.main run publish-marketing --bundle-dir <out_dir>/drop_bundle --out-dir <out_dir>

Tip: add `--drop-id <id>` or `--period-key YYYY-MM-DD` to avoid re-publishing older drafts.
Live publish (optional, requires credentials):
python -m mgc.main run publish-marketing --bundle-dir <out_dir>/drop_bundle --out-dir <out_dir> --publish-live

Live integration env:
- `MGC_PUBLISH_LIVE=1` (or pass `--publish-live`)
- X (Twitter): `MGC_X_API_KEY`, `MGC_X_API_SECRET`, `MGC_X_ACCESS_TOKEN`, `MGC_X_ACCESS_TOKEN_SECRET`
  - If `video_path`/`media_path` is provided, X uploads local media (mp4/gif/jpg/png) before posting.
- YouTube: `MGC_YT_ACCESS_TOKEN` (optional: `MGC_YT_PRIVACY`, `MGC_YT_CATEGORY_ID`)
  - Optional retries: `MGC_YT_RETRY=1`, `MGC_YT_RETRY_MAX=3`, `MGC_YT_RETRY_SLEEP=2`
  - Optional processing poll: `MGC_YT_POLL=0`, `MGC_YT_POLL_MAX=12`, `MGC_YT_POLL_SLEEP=5`
- Instagram Reels: `MGC_IG_ACCESS_TOKEN`, `MGC_IG_USER_ID` + `video_url` in payload (public URL)
  - Processing poll: `MGC_IG_POLL=1` (default), `MGC_IG_POLL_MAX=10`, `MGC_IG_POLL_SLEEP=3`
- TikTok: `MGC_TIKTOK_ACCESS_TOKEN` + `video_url` in payload (public URL)
  - Optional status polling: `MGC_TIKTOK_POLL=1`, `MGC_TIKTOK_POLL_MAX=10`, `MGC_TIKTOK_POLL_SLEEP=3`
- Webhook fallback: `MGC_MARKETING_WEBHOOK_URL` or `MGC_MARKETING_WEBHOOK_<PLATFORM>`
- Webhook media (optional): `MGC_MARKETING_WEBHOOK_INCLUDE_MEDIA=1` to include base64 media payloads
- Media hosting: set `MGC_MARKETING_MEDIA_BASE` to a public URL that serves `marketing/media/` (publish_latest copies it into the web bundle).
  Supports placeholders: `{context}`, `{schedule}`, `{period_key}`, `{drop_id}`, `{track_id}`, `{media_path}`, `{media_file}`, `{marketing_media_path}`.
  If no media placeholders are used, the filename is appended to the base URL.
- Scheduler adapter (credential-less): set `MGC_MARKETING_SCHEDULER_URL` to push all posts to a scheduler service.
  Optional: `MGC_MARKETING_SCHEDULER_AUTH`, `MGC_MARKETING_SCHEDULER_HEADERS` (JSON), `MGC_MARKETING_SCHEDULER_INCLUDE_MEDIA=1`.

---

## Demo Guide

Choose the demo path that matches your environment:

**Offline / local (no external services):**

scripts/verify_all.sh

This runs a deterministic end-to-end check using the fixture DB and stub provider:
music generation, daily/weekly playlists, drop bundle, web build, submission zip,
marketing receipts, and billing CLI.

**Production-style / VM (requires system services):**

sudo -E scripts/demo_check.sh

This command:
- Runs the full daily pipeline
- Publishes `/latest/web/<context>`
- Regenerates `/releases/feed.json`
- Verifies nginx is serving the feed
- Ensures backup contexts are filtered
- Proves deterministic content via stable hashing

If either script exits successfully, the end-to-end flow is valid for that environment.

---

## Observability & Traceability

Artifacts and logs are emitted on every run for auditability:
- `data/evidence/` (drop bundles, evidence JSON, manifests)
- `data/submissions/` (submission.zip + receipts)
- `data/evidence/marketing/receipts/` (marketing publish receipts)
- `logs/` and `artifacts/` (run logs, CI artifacts)

---

## Analytics

The analytics CLI provides reporting over the SQLite DB:

python -m mgc.main analytics overview  
python -m mgc.main analytics tracks --top 5  
python -m mgc.main analytics playlists --limit 5  
python -m mgc.main analytics runs --limit 5  
python -m mgc.main analytics marketing --limit 5  
python -m mgc.main analytics stability  
python -m mgc.main analytics reuse  
python -m mgc.main analytics duration  
python -m mgc.main analytics export overview --format json --out data/analytics/overview.json

---

## Release Feed

MGC exposes a structured release feed describing the latest published web artifacts.

**Production / VM**
- Path: `/var/lib/mgc/releases/feed.json`
- Served at: `http://<host>/releases/feed.json`

**GitHub Pages (Demo)**
- Path: `/releases/feed.json`
- Generated from fixtures using `--stable` mode and deployed alongside the static web demo

### Determinism Guarantee

The feed includes:
- `generated_at` (production only)
- `content_sha256` — a hash of canonicalized content

This allows deterministic verification even when timestamps change.

---

## CI & GitHub Pages

- CI enforces determinism and contract validation
- Web artifacts are built on `main`
- Static output is deployed to GitHub Pages
- No secrets or cloud accounts required

Everything is reviewable directly from GitHub.

---

## Providers

**Stub Provider (Default)**  
Fully deterministic, offline, CI-safe.

export MGC_PROVIDER=stub

**Optional: Riffusion**  
Requires a local inference server.

export MGC_PROVIDER=riffusion  
export MGC_RIFFUSION_URL=http://127.0.0.1:3013/run_inference

**Optional: DiffSinger**  
Use a local command wrapper or an HTTP endpoint.

export MGC_PROVIDER=diffsinger  
export MGC_DIFFSINGER_CMD="python3 scripts/diffsinger_local.py"  
export MGC_DIFFSINGER_SAMPLE_DIR=/path/to/diffsinger/generated/wavs  
export MGC_DIFFSINGER_OUTPUT_FORMAT=wav  
# or: export MGC_DIFFSINGER_ENDPOINT=http://127.0.0.1:8000/generate

Tip: auto-point to the newest DiffSinger output pool:

source scripts/diffsinger_env.sh /Users/admin/ai/DiffSinger/checkpoints/lj_ds_beta6_1213

This helper also sets MGC_PLAYLIST_PROVIDER=any so daily/weekly playlists can mix providers.

---

## Scheduling

Scheduled drops are implemented via GitHub Actions workflows that run on a **self-hosted** runner
(see `.github/workflows/scheduled_drops*.yml`). These jobs only run if the self-hosted runner is online.

If you prefer not to run a self-hosted runner, you can replace this with a cron job that calls
`scripts/run_daily.sh` / `scripts/run_weekly.sh` on an always-on machine.

---

## Why This Project Exists

This repository demonstrates:
- Release engineering discipline
- CI-safe automation
- Deterministic system design
- Explicit approval gates
- Clear separation of concerns
- Auditable, production-grade workflows

---

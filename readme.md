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

**Playlist Builder**
- Deterministic selection and ordering
- Stable playlist JSON

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

### Marketing Publish (Receipts)

Daily/weekly runs emit deterministic post payloads under `marketing/publish/`.  
Publishing is file-based and writes receipts to `marketing/receipts/`.

Environment toggles:
- `MGC_PUBLISH_MARKETING=1` (default) — run `publish-marketing` in `scripts/run_daily.sh`
- `MGC_MARKETING_PLATFORMS="x,tiktok,instagram_reels,youtube_shorts"` — platforms for publish payloads

Manual publish (file mode):
python -m mgc.main run publish-marketing --bundle-dir <out_dir>/drop_bundle --out-dir <out_dir>

Tip: add `--drop-id <id>` or `--period-key YYYY-MM-DD` to avoid re-publishing older drafts.

---

## End-to-End Demo (One Command)

This repository includes a **single command** that proves the entire system end-to-end.

sudo -E scripts/demo_check.sh

This command:
- Runs the full daily pipeline
- Publishes `/latest/web/<context>`
- Regenerates `/releases/feed.json`
- Verifies nginx is serving the feed
- Ensures backup contexts are filtered
- Proves deterministic content via stable hashing

If this script exits successfully, the entire release surface is valid.

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
export RIFFUSION_URL=http://127.0.0.1:3013/run_inference

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

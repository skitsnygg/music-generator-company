# Music Generator Company (MGC)

## Overview

**Music Generator Company (MGC)** is a deterministic, autonomous release pipeline that demonstrates how generated media can be safely built, validated, approved, and published using modern CI/CD and release-engineering practices.

The project is intentionally designed with clear contracts, reproducible artifacts, explicit approval gates, and auditable outputs — without relying on paid APIs or external infrastructure.
**Live Demo (GitHub Pages):**  
https://skitsnygg.github.io/music-generator-company/

---

## System Goals

- End-to-end automation from **generation → release bundle → publish**
- Deterministic, CI-verifiable outputs (byte-for-byte)
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

**Web Player**
- Static HTML + JS
- Built from playlist JSON
- Deployed via GitHub Pages

---

## Release Bundle Layout

out_dir/
- contract_report.json
- drop_evidence.json
- manifest.json
- playlist.json
- tracks/<track_id>.wav
- marketing/receipts/receipts.jsonl
- web/index.html
- web/playlist.json
- web/web_manifest.json
- web/tracks/<track_id>.wav

The bundle is portable, auditable, deployable, and deterministic.

---

## How to Run

### Local Setup

python -m venv .venv  
source .venv/bin/activate  
pip install -U pip  
pip install -e .

No API keys are required for the default stub provider.

---

## End-to-End Demo

### Build a Publish-Ready Release

MGC_PROVIDER=stub python -m mgc.main  
--db fixtures/ci_db.sqlite  
--seed 1  
run autonomous  
--context focus  
--out-dir /tmp/mgc_release  
--deterministic  
--contract publish

---

### Approve the Release

touch /tmp/mgc_release/APPROVED

---

### Publish (Dry Run)

python -m mgc.main run publish  
--bundle-dir /tmp/mgc_release  
--dry-run

---

### Publish (Live)

python -m mgc.main run publish  
--bundle-dir /tmp/mgc_release

---

## CI & GitHub Pages

- CI enforces determinism and contract validation
- Publish bundles are built on `main`
- Static web output is deployed to GitHub Pages
- No secrets or cloud accounts required

Everything is reviewable directly from GitHub.

---

## Providers

**Stub Provider (Default)**
- Fully deterministic
- Offline
- CI-safe

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

## Summary

This is not a demo script — it is a **release pipeline**.

If you can build it, validate it, approve it, and publish it reproducibly, you can ship anything.

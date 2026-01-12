# Music Generator Company (MGC)

## Overview

**Music Generator Company (MGC)** is an autonomous, reproducible pipeline that demonstrates how multiple agents collaborate to generate music, manage metadata, build playlists, and simulate promotion — with full traceability and auditability.

The system focuses on:

- End-to-end automation (generation → storage → playlist → promotion)
- Deterministic, CI-friendly behavior
- Clear observability through logs, JSON artifacts, and receipts
- Clean architecture with explicit agent responsibilities

No paid APIs are required to run the system.

---

## System Goals

- Automated pipeline from **generation → storage → playlist → promotion**
- Context-based music generation (focus / workout / sleep)
- Deterministic outputs suitable for CI verification
- Recurring daily or scheduled music drops
- Clear, auditable proof of promotion via artifacts and receipts

---

## Architecture

The system is organized around **agents**, **orchestration**, and **artifacts**.

### High-Level Architecture Diagram

```
                ┌───────────────────┐
                │  Scheduler / CLI  │
                │ (cron / GH Action │
                │  / manual run)    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Orchestrator    │
                │  mgc run daily    │
                └─────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Music Agent  │   │ Playlist     │   │ Marketing    │
│              │   │ Builder      │   │ Agent        │
│ - generate   │   │ - filter     │   │ - plan posts │
│ - preview    │   │ - shuffle    │   │ - draft JSON │
└──────┬───────┘   │ - dedupe     │   └──────┬───────┘
       │           └──────┬───────┘          │
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Audio Files  │   │ Playlist     │   │ Marketing    │
│ data/tracks  │   │ JSON         │   │ Drafts       │
│ data/previews│   │ data/playlists│  │ data/posts   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────┬───┴──────────────┬───┘
                      ▼                  ▼
              ┌────────────────────────────────┐
              │ Publish Simulation              │
              │ - select DB rows                │
              │ - emit receipts                 │
              └──────────────┬─────────────────┘
                             ▼
                  ┌────────────────────────┐
                  │ Evidence Artifacts      │
                  │ artifacts/runs/         │
                  │ artifacts/receipts/     │
                  └────────────────────────┘
```

### Core Components

#### Agents
- **Music Agent**
  - Generates one track per run (stub or provider-backed)
  - Produces full audio, preview clip, and metadata
- **Marketing Agent**
  - Plans posts for multiple platforms
  - Writes drafts and database rows
- **Billing Agent**
  - MVP stub (included for architectural completeness)

#### Orchestration
- `mgc run daily` coordinates the full pipeline
- CLI subcommands expose each stage independently

#### Storage
- SQLite database (portable, reproducible)
- JSON artifacts written to disk
- Receipts generated for publish simulation

---

## How to Run

### Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

No API keys are required for the default stub provider.

---

## Demo Guide (End-to-End Flow)

### 1. Run the Full Pipeline

```bash
python -m mgc.main run daily --db data/db.sqlite --context focus --json
```

This performs:

- Track generation
- Metadata storage
- Playlist building
- Marketing post planning
- Evidence bundle creation

### 2. Simulate Publishing (Promotion Step)

```bash
python -m mgc.main publish marketing --status published --json
```

This simulates publishing planned posts and generates receipts.

---

## Outputs & Evidence

### Data Outputs (`data/`)

- `data/tracks/` — full audio files
- `data/previews/` — 20-second preview clips
- `data/posts/` — marketing draft payloads
- `data/playlists/<context>_radio.json` — playlist output

### Evidence Bundle (`artifacts/`)

```text
artifacts/
├── runs/<YYYY-MM-DD>/
│   ├── run.json
│   ├── track.json
│   ├── playlist.json
│   └── marketing_posts/
│       ├── <track_id>_x.json
│       ├── <track_id>_tiktok.json
│       └── ...
└── receipts/<YYYY-MM-DD>/marketing/
    ├── x/
    ├── youtube_shorts/
    ├── instagram_reels/
    └── tiktok/
```

These artifacts provide a complete, auditable record of system behavior.

---

## Publish Simulation & Receipts

Publishing is simulated to avoid reliance on live social media APIs.

### What Happens on Publish
- Posts are selected from the database
- Each post is “published” per platform (simulated)
- A receipt JSON is written to disk

Each receipt includes:
- platform
- track_id
- published_at (UTC)
- payload snapshot
- final status
- simulated permalink

This provides reproducible, auditable proof of promotion.

---

## Debug / Trace Mode (Observability)

The system supports **traceable execution** via structured logs and artifacts.

### Logging
- Deterministic UTC timestamps
- No duplicate log handlers
- Log level configurable via CLI or environment variable

```bash
python -m mgc.main run daily --log-level DEBUG --json
```

### Trace Artifacts
During a run, the system emits intermediate artifacts:
- `run.json` — run metadata and configuration
- `track.json` — generated track details
- `playlist.json` — playlist composition
- marketing post drafts
- publish receipts

These allow step-by-step inspection of:
- agent decisions
- intermediate results
- final outputs

### Debug Use Cases
- Inspect why a track was selected for a playlist
- Verify deduplication logic
- Trace marketing payloads before publish
- Compare deterministic rebuilds in CI

---

## Analytics & Observability (Optional)

Analytics are intentionally lightweight.

### What’s Included
- Structured event logging
- CLI-accessible queries (`mgc analytics ...`)
- Deterministic outputs suitable for CI

### Design Intent
- Analytics are **not required** for the core pipeline
- The autonomous pipeline works without analytics enabled
- Analytics support debugging, inspection, and future extensions

Example:
```bash
python -m mgc.main analytics events --limit 20
```

---

## Reproducibility & CI

- Deterministic logging (UTC timestamps)
- Fixture database generator for CI
- Strict rebuild + verify gate

### Run CI Gate Locally

```bash
MGC_DB=fixtures/ci_db.sqlite bash scripts/ci_gate.sh
```

This verifies:
- Python compilation
- Deterministic rebuilds
- Strict output matching

---

## Scheduling (Autonomy)

The project supports recurring autonomous runs.

### GitHub Actions
A scheduled workflow runs:
- `mgc run daily`
- uploads `artifacts/runs/` as build artifacts

### Local Cron Example
```bash
0 6 * * * cd /path/to/repo && . .venv/bin/activate && python -m mgc.main run daily --context focus
```

---

## Providers

### Stub Provider (Default)
- Deterministic
- No external dependencies

```bash
export MGC_PROVIDER=stub
python -m mgc.main run daily --context workout --json
```

### Optional: Riffusion
Requires a running service.

```bash
export MGC_PROVIDER=riffusion
export RIFFUSION_URL="http://127.0.0.1:3013/run_inference"
python -m mgc.main run daily --context sleep --json
```

---

## Database

- SQLite for portability and reproducibility

### Default Paths
- Local: `data/db.sqlite`
- CI: `fixtures/ci_db.sqlite`

### Core Tables
- `tracks`
- `marketing_posts`
- `playlists`
- `playlist_items`

(CI fixtures also include `events`, `playlist_runs`, etc.)

---

## Useful Commands

```bash
# Tracks
python -m mgc.main tracks stats
python -m mgc.main tracks list --limit 10

# Playlists
python -m mgc.main playlists list --limit 10
python -m mgc.main playlists reveal <PLAYLIST_ID>

# Marketing
python -m mgc.main marketing posts list --limit 10
```

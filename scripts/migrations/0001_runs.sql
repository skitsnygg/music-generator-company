-- scripts/migrations/0001_runs.sql
PRAGMA foreign_keys = ON;

-- Canonical runs table (one per run_key)
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,                 -- UUID string
  run_date TEXT NOT NULL,                  -- YYYY-MM-DD
  context TEXT NOT NULL,                   -- e.g. focus/workout/sleep
  seed INTEGER NOT NULL,
  provider_set_version TEXT NOT NULL,      -- e.g. "v1"
  created_at TEXT NOT NULL,                -- UTC ISO
  updated_at TEXT NOT NULL,                -- UTC ISO

  -- Optional metadata fields (nice to have)
  git_sha TEXT,
  git_branch TEXT,
  hostname TEXT,
  argv_json TEXT                            -- JSON string (optional)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_run_key
ON runs(run_date, context, seed, provider_set_version);

-- Stage execution records (for resume + observability later)
CREATE TABLE IF NOT EXISTS run_stages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  stage TEXT NOT NULL,                      -- e.g. "generate_tracks"
  status TEXT NOT NULL,                     -- started|completed|failed
  started_at TEXT,                          -- UTC ISO
  ended_at TEXT,                            -- UTC ISO
  duration_ms INTEGER,
  error_json TEXT,                          -- JSON string if failed
  meta_json TEXT,                           -- JSON string for per-stage outputs

  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_run_stages_unique
ON run_stages(run_id, stage);

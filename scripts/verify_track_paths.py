#!/usr/bin/env python3
"""
scripts/verify_track_paths.py

CI guard: ensure DB tracks.full_path and tracks.preview_path point to real files.

Behavior:
- If a path column is NON-empty -> it MUST exist on disk (else FAIL).
- If a path column is empty -> WARN by default (legacy rows), optional FAIL with --fail-on-empty.

Usage:
  python scripts/verify_track_paths.py --db data/db.sqlite --repo-root .
  python scripts/verify_track_paths.py --db data/db.sqlite --repo-root . --fail-on-empty
  python scripts/verify_track_paths.py --db data/db.sqlite --repo-root . --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _resolve(repo_root: Path, p: str) -> Path:
    rp = Path(p).expanduser()
    if rp.is_absolute():
        return rp
    return (repo_root / rp).resolve()


def _fetch_tracks(con: sqlite3.Connection, limit: int | None) -> List[Tuple[str, str, str]]:
    sql = "SELECT id, full_path, preview_path FROM tracks"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows = con.execute(sql).fetchall()
    out: List[Tuple[str, str, str]] = []
    for r in rows:
        tid = str(r[0])
        fp = "" if r[1] is None else str(r[1])
        pp = "" if r[2] is None else str(r[2])
        out.append((tid, fp, pp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to sqlite DB (e.g. data/db.sqlite)")
    ap.add_argument("--repo-root", default=".", help="Repo root for resolving relative paths")
    ap.add_argument("--limit", type=int, default=0, help="Optional LIMIT for debugging (0 = no limit)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ap.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Fail if any track has empty full_path or preview_path (legacy rows).",
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser()
    repo_root = Path(args.repo_root).expanduser().resolve()
    limit = args.limit if args.limit and args.limit > 0 else None

    if not db_path.exists():
        msg = f"DB not found: {db_path}"
        if args.json:
            print(json.dumps({"ok": False, "reason": "db_missing", "db": str(db_path), "message": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2

    con = sqlite3.connect(str(db_path))
    try:
        tracks = _fetch_tracks(con, limit=limit)
    finally:
        con.close()

    missing: List[Dict[str, Any]] = []
    empty: List[Dict[str, Any]] = []
    checked = 0

    for tid, full_path, preview_path in tracks:
        # full_path
        if _is_nonempty_str(full_path):
            checked += 1
            r = _resolve(repo_root, full_path)
            if not r.is_file():
                missing.append({"id": tid, "col": "full_path", "value": full_path, "resolved": str(r)})
        else:
            empty.append({"id": tid, "col": "full_path", "value": full_path, "resolved": ""})

        # preview_path
        if _is_nonempty_str(preview_path):
            checked += 1
            r = _resolve(repo_root, preview_path)
            if not r.is_file():
                missing.append({"id": tid, "col": "preview_path", "value": preview_path, "resolved": str(r)})
        else:
            empty.append({"id": tid, "col": "preview_path", "value": preview_path, "resolved": ""})

    # We fail only on "missing" by default; empties are warnings unless --fail-on-empty.
    ok = (len(missing) == 0) and (not args.fail_on_empty or len(empty) == 0)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "db": str(db_path),
                    "repo_root": str(repo_root),
                    "track_rows": len(tracks),
                    "paths_checked": checked,
                    "missing_count": len(missing),
                    "empty_count": len(empty),
                    "missing": missing[:200],  # cap output
                    "empty": empty[:200],
                },
                sort_keys=True,
            )
        )
    else:
        if len(empty) > 0:
            print(f"[verify_track_paths] WARN: empty paths={len(empty)} (use --fail-on-empty to make fatal)", file=sys.stderr)
            for e in empty[:20]:
                print(f"  id={e['id']} col={e['col']} value={e['value']!r}", file=sys.stderr)
            if len(empty) > 20:
                print(f"  ... and {len(empty) - 20} more", file=sys.stderr)

        if len(missing) == 0 and (not args.fail_on_empty or len(empty) == 0):
            print(f"[verify_track_paths] OK: tracks={len(tracks)} checked={checked} missing=0 empty={len(empty)}")
        else:
            if len(missing) > 0:
                print(f"[verify_track_paths] FAIL: missing {len(missing)} referenced files", file=sys.stderr)
                for m in missing[:50]:
                    print(
                        f"  id={m['id']} col={m['col']} value={m['value']!r} resolved={m['resolved']!r}",
                        file=sys.stderr,
                    )
                if len(missing) > 50:
                    print(f"  ... and {len(missing) - 50} more", file=sys.stderr)
            if args.fail_on_empty and len(empty) > 0:
                print(f"[verify_track_paths] FAIL: empty paths are fatal (--fail-on-empty)", file=sys.stderr)

    # Exit codes:
    # 0: ok
    # 2: db missing
    # 3: missing referenced files (or empty paths if --fail-on-empty)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

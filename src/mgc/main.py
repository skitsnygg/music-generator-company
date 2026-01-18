#!/usr/bin/env python3
"""
src/mgc/main.py

Music Generator Company CLI

CI compatibility:
- `mgc rebuild ls --json ...`  (CI may put --json after subcommand)
- `mgc rebuild playlists --db ... --out-dir ... --stamp ... --determinism-check --write`
- `mgc rebuild verify playlists ...`
- `mgc rebuild tracks ...`
- `mgc rebuild verify tracks ...`

Other commands:
- status: quick health + recent activity summary
- playlists: list, history, reveal, export
- tracks: list, show, stats
- marketing: posts list + receipts list (file-based publish audit trail)
- providers: list/check provider configuration (env-driven)
- analytics: delegated to mgc.analytics_cli if available
- run: autonomous pipeline entrypoint (daily/weekly)
- web: static web player build/serve
- submission: build + verify deterministic submission ZIPs

Logging:
- deterministic UTC timestamps
- no duplicate logging
- if --log-file is set, default file-only logs (use --log-console to also log to stderr)
- JSON mode disables console logging to keep stdout clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple

from mgc.submission_cli import register_submission_subcommand

DEFAULT_DB = "data/db.sqlite"
DEFAULT_PLAYLIST_DIR = Path("data/playlists")
DEFAULT_TRACKS_DIR = Path("data/tracks")
DEFAULT_TRACKS_EXPORT = DEFAULT_TRACKS_DIR / "tracks.json"

LOG_FMT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


# ----------------------------
# basic utils
# ----------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> NoReturn:
    eprint(msg)
    raise SystemExit(code)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def read_json_file(path: Path) -> Any:
    return json.loads(read_text(path))


def stable_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_file(path: Path, obj: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Read a JSONL file safely. Skips blank lines.
    Returns dict objects; non-dict JSON lines are wrapped as {"value": ...}.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                else:
                    yield {"value": obj}
            except Exception:
                # Keep it resilient: surface raw line
                yield {"raw": s}


# ----------------------------
# argv preprocessing (make global flags work anywhere)
# ----------------------------

_GLOBAL_FLAG_NO_VALUE = {
    "--json",
    "--log-console",
}

_GLOBAL_FLAG_WITH_VALUE = {
    "--db",
    "--log-level",
    "--log-file",
}


def _split_eq(arg: str) -> Tuple[str, Optional[str]]:
    # supports --db=path style
    if arg.startswith("--") and "=" in arg:
        k, v = arg.split("=", 1)
        return k, v
    return arg, None


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """
    Allow global flags to appear anywhere in argv (before or after subcommands)
    by hoisting them to the front before argparse parses.

    This is required because CI may invoke:
      mgc run daily --repo-root /path ...

    Without hoisting, argparse treats --repo-root as a subcommand arg and errors.
    """

    flags_with_value = {
        "--db",
        "--repo-root",
        "--log-level",
        "--log-file",
        "--seed",
    }

    flags_no_value = {
        "--log-console",
        "--json",
        "--no-resume",
    }

    hoisted: list[str] = []
    rest: list[str] = []

    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]

        # Handle --flag=value form
        if tok.startswith("--") and "=" in tok:
            name, _val = tok.split("=", 1)
            if name in flags_with_value or name in flags_no_value:
                hoisted.append(tok)
            else:
                rest.append(tok)
            i += 1
            continue

        # Handle flags that take a value: --flag VALUE
        if tok in flags_with_value:
            hoisted.append(tok)
            if i + 1 < n:
                hoisted.append(argv[i + 1])
                i += 2
            else:
                # Let argparse complain later; keep shape stable
                i += 1
            continue

        # Handle boolean flags: --flag
        if tok in flags_no_value:
            hoisted.append(tok)
            i += 1
            continue

        # Otherwise keep in place
        rest.append(tok)
        i += 1

    return hoisted + rest


# ----------------------------
# logging (deterministic, no duplicates)
# ----------------------------

class _UTCFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def _parse_level(level: str) -> int:
    try:
        return int(level)
    except Exception:
        pass
    lvl = logging.getLevelName(str(level).upper())
    return lvl if isinstance(lvl, int) else logging.INFO


def _clear_all_non_root_handlers() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    mgr = logging.root.manager
    for name, logger_obj in list(mgr.loggerDict.items()):
        if not isinstance(logger_obj, logging.Logger):
            continue
        if name == "root":
            continue
        for h in list(logger_obj.handlers):
            logger_obj.removeHandler(h)
        logger_obj.propagate = True


def _configure_logging(
    *,
    level: str,
    log_file: Optional[str],
    log_console: bool,
    json_mode: bool = False,
) -> None:
    """
    Logging policy:
    - Always log to file if --log-file is set.
    - Console logs go to STDERR (never stdout).
    - If json_mode is True, do NOT log to console at all (keep stdout pure JSON).
    - If no log_file is set, default to console stderr.
    - If log_file is set, console stderr is only enabled with --log-console.
    """
    _clear_all_non_root_handlers()

    fmt = _UTCFormatter(LOG_FMT, datefmt=LOG_DATEFMT)
    handlers: List[logging.Handler] = []

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        handlers.append(fh)

    if not json_mode:
        if (not log_file) or log_console:
            sh = logging.StreamHandler(stream=sys.stderr)
            sh.setFormatter(fmt)
            handlers.append(sh)

    if not handlers:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setFormatter(fmt)
        handlers.append(sh)

    logging.basicConfig(
        level=_parse_level(level),
        handlers=handlers,
        force=True,
    )


# ----------------------------
# Providers (env-driven; run_cli owns actual selection)
# ----------------------------

_SUPPORTED_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "stub": {
        "desc": "Deterministic built-in stub generator (CI friendly).",
        "env": [],
    },
    "filesystem": {
        "desc": "Pick an audio file from a directory pool and copy it into out_dir/tracks/<track_id>.wav.",
        "env": ["MGC_FS_PROVIDER_DIR"],
        "notes": [
            "Pool is scanned recursively.",
            "Prefer .wav if any exist; otherwise uses any supported audio file.",
        ],
    },
}

_AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff")


def _provider_from_env() -> str:
    return (os.environ.get("MGC_PROVIDER") or "stub").strip().lower() or "stub"


def _scan_audio_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _AUDIO_SUFFIXES:
            out.append(p)
    return sorted(out)


def cmd_providers_list(args: argparse.Namespace) -> int:
    cur = _provider_from_env()
    items = []
    for name in sorted(_SUPPORTED_PROVIDERS.keys()):
        meta = dict(_SUPPORTED_PROVIDERS[name])
        meta["name"] = name
        meta["selected"] = (name == cur)
        items.append(meta)

    if args.json:
        print(json.dumps({"selected": cur, "providers": items}, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    print(f"selected: {cur}")
    for it in items:
        mark = "*" if it.get("selected") else " "
        print(f"{mark} {it['name']}: {it.get('desc','')}")
        env = it.get("env") or []
        if env:
            print(f"    env: {', '.join(env)}")
        notes = it.get("notes") or []
        for n in notes:
            print(f"    note: {n}")
    return 0


def cmd_providers_check(args: argparse.Namespace) -> int:
    """
    Validate env config for the currently selected provider (via MGC_PROVIDER).
    This does NOT run the pipeline; it just checks that the provider can be instantiated/configured.
    """
    sel = _provider_from_env()
    info = _SUPPORTED_PROVIDERS.get(sel)

    out: Dict[str, Any] = {
        "selected": sel,
        "ok": True,
        "problems": [],
        "env": {
            "MGC_PROVIDER": os.environ.get("MGC_PROVIDER"),
            "MGC_FS_PROVIDER_DIR": os.environ.get("MGC_FS_PROVIDER_DIR"),
        },
    }

    if info is None:
        out["ok"] = False
        out["problems"].append(f"unknown_provider:{sel}")
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            print(f"FAIL: unknown provider: {sel}")
            print("known:", ", ".join(sorted(_SUPPORTED_PROVIDERS.keys())))
        return 2

    if sel == "filesystem":
        d = (os.environ.get("MGC_FS_PROVIDER_DIR") or "").strip()
        if not d:
            out["ok"] = False
            out["problems"].append("missing_env:MGC_FS_PROVIDER_DIR")
        else:
            root = Path(d).expanduser().resolve()
            out["filesystem"] = {"root_dir": str(root), "exists": root.exists(), "is_dir": root.is_dir()}
            files = _scan_audio_files(root) if (root.exists() and root.is_dir()) else []
            out["filesystem"]["file_count"] = len(files)
            if files:
                # show a couple examples for sanity
                out["filesystem"]["examples"] = [str(p) for p in files[:3]]
            if not (root.exists() and root.is_dir()):
                out["ok"] = False
                out["problems"].append("invalid_dir:MGC_FS_PROVIDER_DIR")
            elif not files:
                out["ok"] = False
                out["problems"].append("no_audio_files_found")

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        if out["ok"]:
            print(f"OK: provider={sel}")
            if sel == "filesystem":
                fs = out.get("filesystem", {})
                print(f"  root_dir: {fs.get('root_dir')}")
                print(f"  files: {fs.get('file_count')}")
                for ex in (fs.get("examples") or []):
                    print(f"    ex: {ex}")
        else:
            print(f"FAIL: provider={sel}")
            for p in out["problems"]:
                print(f"  - {p}")
            if sel == "filesystem":
                print('hint: export MGC_FS_PROVIDER_DIR=/Users/admin/music-generator-company/data/tracks')
                print("hint: ensure the directory contains .wav/.mp3/.m4a/etc files")

    return 0 if out["ok"] else 2

# ----------------------------
# db migrate/status (run migrations from anywhere)
# ----------------------------

def _infer_repo_root(repo_root_arg: str | None) -> Path:
    if repo_root_arg:
        return Path(repo_root_arg).expanduser().resolve()
    # src/mgc/main.py -> repo root is two parents up from src/
    return Path(__file__).resolve().parents[2]


def _migrations_dir(repo_root: Path) -> Path:
    return (repo_root / "scripts" / "migrations").resolve()


def _ensure_schema_migrations(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )


def _applied_migration_versions(con: sqlite3.Connection) -> set[str]:
    _ensure_schema_migrations(con)
    rows = con.execute("SELECT version FROM schema_migrations ORDER BY version ASC").fetchall()
    return {str(r[0]) for r in rows}


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _migration_now_iso(args_now: str | None) -> str:
    if args_now:
        s = args_now.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    return _utc_now_iso()


def cmd_db_migrate(args: argparse.Namespace) -> int:
    repo_root = _infer_repo_root(getattr(args, "repo_root", None))
    db_path = Path(args.db).expanduser().resolve()
    mig_dir = Path(args.migrations_dir).expanduser().resolve() if args.migrations_dir else _migrations_dir(repo_root)

    if not db_path.exists():
        die(f"DB not found: {db_path}")
    if not mig_dir.exists() or not mig_dir.is_dir():
        die(f"migrations dir missing: {mig_dir}")

    sql_files = sorted([p for p in mig_dir.glob("*.sql") if p.is_file()])

    applied_count = 0
    applied: list[str] = []
    skipped: list[str] = []

    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON;")
        _ensure_schema_migrations(con)
        already = _applied_migration_versions(con)

        now_ts = _migration_now_iso(getattr(args, "now", None))

        con.execute("BEGIN;")
        try:
            for f in sql_files:
                version = f.name
                if version in already:
                    skipped.append(version)
                    continue

                sql = _read_sql(f).strip()
                if sql:
                    con.executescript(sql)

                con.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                    (version, now_ts),
                )
                applied.append(version)
                applied_count += 1

            con.commit()
        except Exception:
            con.rollback()
            raise

    finally:
        con.close()

    out = {
        "ok": True,
        "cmd": "db.migrate",
        "db": str(db_path),
        "migrations_dir": str(mig_dir),
        "applied_count": applied_count,
        "applied": applied,
        "skipped_count": len(skipped),
    }

    if getattr(args, "json", False):
        print(stable_dumps(out))
    else:
        print(f"OK: applied {applied_count} migrations to {db_path}")
    return 0


def cmd_db_status(args: argparse.Namespace) -> int:
    repo_root = _infer_repo_root(getattr(args, "repo_root", None))
    db_path = Path(args.db).expanduser().resolve()
    mig_dir = Path(args.migrations_dir).expanduser().resolve() if args.migrations_dir else _migrations_dir(repo_root)

    if not db_path.exists():
        die(f"DB not found: {db_path}")

    expected = sorted([p.name for p in mig_dir.glob("*.sql")]) if mig_dir.exists() else []

    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        _ensure_schema_migrations(con)
        rows = con.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version ASC").fetchall()
        applied_versions = [str(r["version"]) for r in rows]
        applied_set = set(applied_versions)

        missing = [v for v in expected if v not in applied_set]

        billing_tables = {
            "billing_users": _table_exists(con, "billing_users"),
            "billing_tokens": _table_exists(con, "billing_tokens"),
            "billing_entitlements": _table_exists(con, "billing_entitlements"),
        }

    finally:
        con.close()

    out = {
        "ok": True,
        "cmd": "db.status",
        "db": str(db_path),
        "migrations_dir": str(mig_dir),
        "expected_count": len(expected),
        "applied_count": len(applied_versions),
        "applied": rows and [{"version": str(r["version"]), "applied_at": str(r["applied_at"])} for r in rows] or [],
        "missing": missing,
        "billing_tables": billing_tables,
    }

    if getattr(args, "json", False):
        print(stable_dumps(out))
    else:
        print(f"DB: {db_path}")
        print(f"migrations_dir: {mig_dir}")
        print(f"expected: {len(expected)}  applied: {len(applied_versions)}  missing: {len(missing)}")
        if missing:
            for m in missing:
                print(f"  missing: {m}")
        for k, v in billing_tables.items():
            print(f"{k}: {'YES' if v else 'NO'}")
    return 0

# ----------------------------
# DB helpers
# ----------------------------

@dataclass(frozen=True)
class DBConn:
    path: Path

    def connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError(f"DB not found: {self.path}")
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _maybe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _safe_select_playlist_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    slug: Optional[str] = None,
) -> List[sqlite3.Row]:
    if not _table_exists(conn, "playlists"):
        return []

    cols = set(_columns(conn, "playlists"))

    order_by = None
    if "created_at" in cols:
        order_by = "created_at DESC"
    elif "created_ts" in cols:
        order_by = "created_ts DESC"
    elif "id" in cols:
        order_by = "id DESC"

    where = ""
    params: List[Any] = []
    if slug and "slug" in cols:
        where = "WHERE slug = ?"
        params.append(slug)

    q = "SELECT * FROM playlists "
    if where:
        q += where + " "
    if order_by:
        q += f"ORDER BY {order_by} "
    q += "LIMIT ?"
    params.append(limit)

    return conn.execute(q, tuple(params)).fetchall()


def _safe_select_latest_playlists_by_slug(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    if not _table_exists(conn, "playlists"):
        return []

    cols = set(_columns(conn, "playlists"))
    if "slug" not in cols:
        return _safe_select_playlist_rows(conn, limit=1_000_000)

    ts_col = "created_at" if "created_at" in cols else ("created_ts" if "created_ts" in cols else None)

    if ts_col:
        q = f"""
        SELECT p.*
        FROM playlists p
        JOIN (
          SELECT slug, MAX({ts_col}) AS mx
          FROM playlists
          GROUP BY slug
        ) x
        ON p.slug = x.slug AND p.{ts_col} = x.mx
        ORDER BY p.slug ASC
        """
    else:
        q = """
        SELECT p.*
        FROM playlists p
        JOIN (
          SELECT slug, MAX(id) AS mx
          FROM playlists
          GROUP BY slug
        ) x
        ON p.slug = x.slug AND p.id = x.mx
        ORDER BY p.slug ASC
        """
    return conn.execute(q).fetchall()


def _safe_get_playlist_by_id(conn: sqlite3.Connection, playlist_id: str) -> Optional[sqlite3.Row]:
    if not _table_exists(conn, "playlists"):
        return None
    cols = set(_columns(conn, "playlists"))
    if "id" not in cols:
        return None
    return conn.execute("SELECT * FROM playlists WHERE id = ? LIMIT 1", (playlist_id,)).fetchone()


def _safe_select_playlist_items(conn: sqlite3.Connection, playlist_id: str, limit: int) -> List[sqlite3.Row]:
    if not _table_exists(conn, "playlist_items"):
        return []
    cols = set(_columns(conn, "playlist_items"))
    if "playlist_id" not in cols:
        return []

    q = "SELECT * FROM playlist_items WHERE playlist_id = ?"
    params: List[Any] = [playlist_id]

    if "position" in cols:
        q += " ORDER BY position ASC"
    elif "idx" in cols:
        q += " ORDER BY idx ASC"
    elif "created_at" in cols:
        q += " ORDER BY created_at ASC"
    q += " LIMIT ?"
    params.append(limit)

    return conn.execute(q, tuple(params)).fetchall()


def _safe_get_track_by_id(conn: sqlite3.Connection, track_id: str) -> Optional[sqlite3.Row]:
    if not _table_exists(conn, "tracks"):
        return None
    cols = set(_columns(conn, "tracks"))
    if "id" not in cols:
        return None
    return conn.execute("SELECT * FROM tracks WHERE id = ? LIMIT 1", (track_id,)).fetchone()


# ----------------------------
# formatting
# ----------------------------

def _fmt_playlist_list_row(row: sqlite3.Row) -> str:
    d = _row_to_dict(row)
    pid = str(d.get("id", ""))
    slug = str(d.get("slug", d.get("name", "")) or "")
    created = d.get("created_at", d.get("created_ts", d.get("created", "")))

    tracks_n = None
    for key in ("tracks", "track_count", "num_tracks", "n_tracks"):
        if key in d:
            tracks_n = _maybe_int(d.get(key))
            break

    json_path = d.get("json_path", d.get("path", ""))

    parts = [pid, slug]
    if created:
        parts.append(str(created))
    if tracks_n is not None:
        parts.append(f"tracks={tracks_n}")
    if json_path:
        parts.append(f"json_path={json_path}")
    return "  " + "  ".join(parts)


# ----------------------------
# playlist JSON extraction/rebuild
# ----------------------------

def _extract_playlist_json_from_row(d: Dict[str, Any]) -> Optional[Any]:
    for key in ("json", "playlist_json", "payload", "payload_json", "data", "data_json", "json_blob"):
        if key not in d:
            continue
        val = d.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            return val
        if isinstance(val, (bytes, bytearray)):
            try:
                return json.loads(val.decode("utf-8"))
            except Exception:
                continue
        if isinstance(val, str):
            s = val.strip()
            if not s:
                continue
            try:
                return json.loads(s)
            except Exception:
                continue
    return None


def _reconstruct_playlist_json(
    conn: sqlite3.Connection,
    playlist_row: Dict[str, Any],
    *,
    item_limit: int = 500,
) -> Dict[str, Any]:
    pid = str(playlist_row.get("id", ""))
    slug = str(playlist_row.get("slug", playlist_row.get("name", "")) or "")
    created_at = playlist_row.get("created_at", playlist_row.get("created_ts", playlist_row.get("created", "")))

    items = _safe_select_playlist_items(conn, pid, limit=item_limit)
    item_dicts = [_row_to_dict(r) for r in items]

    tracks: List[Dict[str, Any]] = []
    for it in item_dicts:
        tid = None
        for k in ("track_id", "track_uuid", "track", "id_track"):
            if k in it and it[k]:
                tid = str(it[k])
                break
        if tid:
            tr = _safe_get_track_by_id(conn, tid)
            if tr is not None:
                tracks.append(_row_to_dict(tr))
                continue
        tracks.append(it)

    return {"id": pid, "slug": slug, "created_at": created_at, "tracks": tracks}


def _safe_slug(s: str) -> str:
    s2 = s.strip().replace(" ", "_")
    return s2 if s2 else "playlist"


def _build_playlist_json_for_row(conn: sqlite3.Connection, d: Dict[str, Any]) -> Any:
    src = d.get("json_path") or d.get("path")
    if src:
        src_path = Path(str(src))
        if src_path.exists():
            try:
                return read_json_file(src_path)
            except Exception:
                pass

    blob = _extract_playlist_json_from_row(d)
    if blob is not None:
        return blob

    return _reconstruct_playlist_json(conn, d)

def _safe_get_account(conn, account_id: str) -> Optional[sqlite3.Row]:
    if not _table_exists(conn, "billing_accounts"):
        return None
    return conn.execute(
        "SELECT * FROM billing_accounts WHERE id = ? LIMIT 1",
        (account_id,)
    ).fetchone()

def _safe_has_entitlement(conn, account_id: str, entitlement: str) -> bool:
    if not _table_exists(conn, "billing_entitlements"):
        return False
    row = conn.execute(
        "SELECT 1 FROM billing_entitlements WHERE account_id = ? AND entitlement = ?",
        (account_id, entitlement)
    ).fetchone()
    return row is not None


# ----------------------------
# playlists/tracks/marketing commands
# ----------------------------

def cmd_playlists_list(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.playlists.list")
    db = DBConn(Path(args.db))
    with db.connect() as conn:
        rows = _safe_select_playlist_rows(conn, limit=args.limit, slug=args.slug)

    if args.json:
        print(json.dumps([_row_to_dict(r) for r in rows], indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not rows:
        log.info("No playlists found.")
        return 0

    for r in rows:
        print(_fmt_playlist_list_row(r))
    return 0


def cmd_playlists_history(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.playlists.history")
    db = DBConn(Path(args.db))
    target = args.target

    with db.connect() as conn:
        by_id = _safe_get_playlist_by_id(conn, target)
        if by_id is not None:
            d = _row_to_dict(by_id)
            slug = d.get("slug")
            rows = _safe_select_playlist_rows(conn, limit=args.limit, slug=slug) if slug else [by_id]
        else:
            rows = _safe_select_playlist_rows(conn, limit=args.limit, slug=target)

    if args.json:
        print(json.dumps([_row_to_dict(r) for r in rows], indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not rows:
        log.info("No playlist history found for: %s", target)
        return 0

    for i, r in enumerate(rows, start=1):
        print(f"{i:2d}. {_fmt_playlist_list_row(r).strip()}")
    return 0


def cmd_playlists_reveal(args: argparse.Namespace) -> int:
    db = DBConn(Path(args.db))
    playlist_id = args.id
    json_path: Optional[Path] = None

    with db.connect() as conn:
        row = _safe_get_playlist_by_id(conn, playlist_id)
        if row is not None:
            d = _row_to_dict(row)
            p = d.get("json_path") or d.get("path")
            if p:
                json_path = Path(str(p))

    if json_path is None:
        candidate = Path(playlist_id)
        if candidate.exists() and candidate.is_file():
            json_path = candidate

    if json_path is None:
        die(f"Could not find playlist JSON for id/path: {playlist_id}")

    data = read_json_file(json_path)
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_playlists_export(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.playlists.export")
    db = DBConn(Path(args.db))
    out_dir = Path(args.out_dir)

    exported: List[str] = []
    with db.connect() as conn:
        rows = _safe_select_playlist_rows(conn, limit=args.limit, slug=args.slug)
        if not rows:
            log.info("No playlists to export.")
            return 0

        for r in rows:
            d = _row_to_dict(r)
            pid = str(d.get("id", ""))
            slug = str(d.get("slug", "")) if d.get("slug") else ""
            safe = _safe_slug(slug) if slug else "playlist"
            dst = out_dir / (f"{safe}_{pid}.json" if pid else f"{safe}.json")

            obj = _build_playlist_json_for_row(conn, d)
            write_json_file(dst, obj)
            exported.append(str(dst))

    if args.json:
        print(json.dumps({"exported": exported}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for pth in exported:
            print(pth)
    return 0


def cmd_tracks_list(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.tracks.list")
    db = DBConn(Path(args.db))

    with db.connect() as conn:
        if not _table_exists(conn, "tracks"):
            if args.json:
                print("[]")
            else:
                log.info("No tracks table found.")
            return 0

        cols = set(_columns(conn, "tracks"))
        where: List[str] = []
        params: List[Any] = []

        if args.mood and "mood" in cols:
            where.append("mood = ?")
            params.append(args.mood)
        if args.genre and "genre" in cols:
            where.append("genre = ?")
            params.append(args.genre)

        q = "SELECT * FROM tracks "
        if where:
            q += "WHERE " + " AND ".join(where) + " "
        if "created_at" in cols:
            q += "ORDER BY created_at DESC "
        elif "id" in cols:
            q += "ORDER BY id DESC "
        q += "LIMIT ?"
        params.append(args.limit)

        rows = conn.execute(q, tuple(params)).fetchall()

    if args.json:
        print(json.dumps([_row_to_dict(r) for r in rows], indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not rows:
        log.info("No tracks found.")
        return 0

    for r in rows:
        d = _row_to_dict(r)
        parts = [str(d.get("id", ""))]
        for key in ("title", "name", "slug"):
            if d.get(key):
                parts.append(str(d[key]))
                break
        if d.get("mood"):
            parts.append(f"mood={d['mood']}")
        if d.get("genre"):
            parts.append(f"genre={d['genre']}")
        print("  " + "  ".join(parts))
    return 0


def cmd_tracks_show(args: argparse.Namespace) -> int:
    db = DBConn(Path(args.db))
    with db.connect() as conn:
        if not _table_exists(conn, "tracks"):
            die("No tracks table found.")
        cols = set(_columns(conn, "tracks"))
        if "id" not in cols:
            die("tracks table has no id column.")
        row = conn.execute("SELECT * FROM tracks WHERE id = ? LIMIT 1", (args.id,)).fetchone()

    if row is None:
        die(f"Track not found: {args.id}")

    print(json.dumps(_row_to_dict(row), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_tracks_stats(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.tracks.stats")
    db = DBConn(Path(args.db))

    with db.connect() as conn:
        if not _table_exists(conn, "tracks"):
            if args.json:
                print("{}")
            else:
                log.info("No tracks table found.")
            return 0

        cols = set(_columns(conn, "tracks"))
        total = int(conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"])

        by_mood: Dict[str, int] = {}
        if "mood" in cols:
            rows = conn.execute("SELECT mood, COUNT(*) AS n FROM tracks GROUP BY mood ORDER BY mood ASC").fetchall()
            by_mood = {str(r["mood"]): int(r["n"]) for r in rows}

        by_genre: Dict[str, int] = {}
        if "genre" in cols:
            rows = conn.execute("SELECT genre, COUNT(*) AS n FROM tracks GROUP BY genre ORDER BY genre ASC").fetchall()
            by_genre = {str(r["genre"]): int(r["n"]) for r in rows}

    out = {"total": total, "by_mood": by_mood, "by_genre": by_genre}
    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_marketing_posts_list(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.marketing.posts.list")
    db = DBConn(Path(args.db))

    with db.connect() as conn:
        if not _table_exists(conn, "marketing_posts"):
            if args.json:
                print("[]")
            else:
                log.info("No marketing_posts table found.")
            return 0

        cols = set(_columns(conn, "marketing_posts"))
        q = "SELECT * FROM marketing_posts "
        if "created_at" in cols:
            q += "ORDER BY created_at DESC "
        elif "id" in cols:
            q += "ORDER BY id DESC "
        q += "LIMIT ?"

        rows = conn.execute(q, (args.limit,)).fetchall()

    if args.json:
        print(json.dumps([_row_to_dict(r) for r in rows], indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not rows:
        log.info("No marketing posts found.")
        return 0

    for r in rows:
        d = _row_to_dict(r)
        pid = d.get("id", "")
        title = d.get("title", d.get("slug", "")) or ""
        created = d.get("created_at", "")
        print(f"  {pid}  {title}  {created}".rstrip())
    return 0


def cmd_marketing_receipts_list(args: argparse.Namespace) -> int:
    """
    List publish receipts from a JSONL file written by the publish step.

    This is intentionally file-based (not DB-based) so it stays:
      - portable
      - deterministic-friendly
      - easy to audit
    """
    log = logging.getLogger("mgc.marketing.receipts.list")
    path = Path(args.path)

    if not path.exists():
        if args.json:
            print(json.dumps({"path": str(path), "exists": False, "receipts": []}, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            log.info("receipts file missing: %s", path)
        return 0

    rows = list(_iter_jsonl(path))
    # newest last (append-only), so take from the end
    lim = int(args.limit)
    picked = rows[-lim:] if lim > 0 else rows

    if args.json:
        out = {
            "path": str(path),
            "exists": True,
            "sha256": _sha256_file(path),
            "count": len(rows),
            "shown": len(picked),
            "receipts": picked,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    print(f"receipts: {path}  (count={len(rows)} sha256={_sha256_file(path)})")
    for r in picked:
        platform = r.get("platform") or r.get("channel") or ""
        post_id = r.get("post_id") or r.get("id") or ""
        status = r.get("status") or ""
        ts = r.get("published_ts") or r.get("ts") or ""
        drop_id = r.get("drop_id") or ""
        print(f"  {ts}  {platform}  {status}  post_id={post_id}  drop_id={drop_id}".rstrip())
    return 0


# ----------------------------
# status command
# ----------------------------

def _db_count(conn: sqlite3.Connection, table: str) -> Optional[int]:
    if not _table_exists(conn, table):
        return None
    try:
        return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
    except Exception:
        return None


def _safe_latest_row(conn: sqlite3.Connection, table: str) -> Optional[sqlite3.Row]:
    if not _table_exists(conn, table):
        return None
    cols = set(_columns(conn, table))
    order_by = None
    if "created_at" in cols:
        order_by = "created_at DESC"
    elif "created_ts" in cols:
        order_by = "created_ts DESC"
    elif "id" in cols:
        order_by = "id DESC"
    q = f"SELECT * FROM {table} "
    if order_by:
        q += f"ORDER BY {order_by} "
    q += "LIMIT 1"
    try:
        return conn.execute(q).fetchone()
    except Exception:
        return None


def _path_info(path: Path) -> Dict[str, Any]:
    try:
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        is_file = path.is_file() if exists else False
        size = path.stat().st_size if (exists and is_file) else None
        mtime = (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            if exists
            else None
        )
        return {
            "path": str(path),
            "exists": exists,
            "is_dir": is_dir,
            "is_file": is_file,
            "size": size,
            "mtime_utc": mtime,
        }
    except Exception as e:
        return {"path": str(path), "exists": False, "error": str(e)}


def cmd_status(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    out: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "db": {"path": str(db_path), "exists": db_path.exists()},
        "env": {
            "MGC_DB": os.environ.get("MGC_DB"),
            "MGC_CONTEXT": os.environ.get("MGC_CONTEXT"),
            "MGC_SEED": os.environ.get("MGC_SEED"),
            "MGC_PROVIDER": os.environ.get("MGC_PROVIDER"),
            "MGC_DETERMINISTIC": os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC"),
            "MGC_ARTIFACTS_DIR": os.environ.get("MGC_ARTIFACTS_DIR"),
            "MGC_FS_PROVIDER_DIR": os.environ.get("MGC_FS_PROVIDER_DIR"),
        },
        "tables": {},
        "latest": {},
        "paths": {},
        "notes": [],
    }

    out["paths"]["playlists_dir"] = _path_info(DEFAULT_PLAYLIST_DIR)
    out["paths"]["tracks_dir"] = _path_info(DEFAULT_TRACKS_DIR)
    out["paths"]["tracks_export"] = _path_info(DEFAULT_TRACKS_EXPORT)
    out["paths"]["evidence_dir"] = _path_info(Path("data/evidence"))

    # File-based publish receipts (default location used by run_cli)
    out["paths"]["marketing_receipts"] = _path_info(Path("artifacts/run/marketing/receipts.jsonl"))

    if not db_path.exists():
        out["notes"].append("db_missing")
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
            return 0

        print("MGC Status")
        print(f"  time_utc: {_utc_now_iso()}")
        print(f"  db: {db_path} (MISSING)")
        print("  (set --db or MGC_DB)")
        return 0

    db = DBConn(db_path)
    with db.connect() as conn:
        for t in ("events", "playlist_runs", "playlists", "playlist_items", "tracks", "marketing_posts"):
            out["tables"][t] = {"exists": _table_exists(conn, t), "count": _db_count(conn, t)}

        latest_playlist = _safe_latest_row(conn, "playlists")
        if latest_playlist is not None:
            out["latest"]["playlist"] = _row_to_dict(latest_playlist)

        latest_track = _safe_latest_row(conn, "tracks")
        if latest_track is not None:
            out["latest"]["track"] = _row_to_dict(latest_track)

        latest_post = _safe_latest_row(conn, "marketing_posts")
        if latest_post is not None:
            out["latest"]["marketing_post"] = _row_to_dict(latest_post)

        latest_event = _safe_latest_row(conn, "events")
        if latest_event is not None:
            out["latest"]["event"] = _row_to_dict(latest_event)

        latest_run = _safe_latest_row(conn, "playlist_runs")
        if latest_run is not None:
            out["latest"]["playlist_run"] = _row_to_dict(latest_run)

        try:
            rows = _safe_select_latest_playlists_by_slug(conn)
            out["latest"]["playlists_by_slug"] = [
                {
                    "id": item.get("id"),
                    "slug": item.get("slug", item.get("name")),
                    "created_at": item.get("created_at", item.get("created_ts")),
                }
                for item in (_row_to_dict(x) for x in rows)
            ]
        except Exception:
            pass

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    print("MGC Status")
    print(f"  time_utc: {out['ts_utc']}")
    print(f"  db: {out['db']['path']} ({'OK' if out['db']['exists'] else 'MISSING'})")

    env_items = out["env"]
    env_show = []
    for k in ("MGC_CONTEXT", "MGC_SEED", "MGC_PROVIDER", "MGC_DETERMINISTIC", "MGC_ARTIFACTS_DIR", "MGC_FS_PROVIDER_DIR"):
        v = env_items.get(k)
        if v:
            env_show.append(f"{k}={v}")
    if env_show:
        print("  env:")
        for line in env_show:
            print(f"    {line}")

    print("  tables:")
    for t in ("events", "playlist_runs", "playlists", "playlist_items", "tracks", "marketing_posts"):
        info = out["tables"].get(t, {})
        exists = info.get("exists", False)
        cnt = info.get("count", None)
        cnt_s = str(cnt) if cnt is not None else "?"
        print(f"    {t}: {'present' if exists else 'missing'}  count={cnt_s}")

    print("  paths:")
    for key in ("playlists_dir", "tracks_dir", "tracks_export", "evidence_dir", "marketing_receipts"):
        p = out["paths"].get(key, {})
        status = "OK" if p.get("exists") else "MISSING"
        extra = ""
        if p.get("is_file") and p.get("size") is not None:
            extra = f" size={p['size']}"
        if p.get("mtime_utc"):
            extra += f" mtime_utc={p['mtime_utc']}"
        print(f"    {key}: {p.get('path')} ({status}){extra}")

    latest = out.get("latest", {})
    if latest:
        print("  latest:")
        if "playlist" in latest:
            d = latest["playlist"]
            print(
                f"    playlist: id={d.get('id')} slug={d.get('slug', d.get('name'))} created={d.get('created_at', d.get('created_ts'))}"
            )
        if "track" in latest:
            d = latest["track"]
            title = d.get("title") or d.get("name") or d.get("slug") or ""
            print(f"    track: id={d.get('id')} {title}".rstrip())
        if "marketing_post" in latest:
            d = latest["marketing_post"]
            title = d.get("title") or d.get("slug") or ""
            print(f"    marketing_post: id={d.get('id')} {title}".rstrip())
        if "playlist_run" in latest:
            d = latest["playlist_run"]
            created = d.get("created_at", d.get("created_ts"))
            print(f"    playlist_run: id={d.get('id')} created={created}".rstrip())
        if "event" in latest:
            d = latest["event"]
            kind = d.get("kind") or d.get("type") or d.get("name") or ""
            created = d.get("created_at", d.get("created_ts"))
            print(f"    event: id={d.get('id')} {kind} created={created}".rstrip())

        pbs = latest.get("playlists_by_slug")
        if isinstance(pbs, list) and pbs:
            print("    playlists_by_slug:")
            for item in pbs[:25]:
                print(f"      {item.get('slug')}  id={item.get('id')}  created={item.get('created_at')}")
            if len(pbs) > 25:
                print(f"      ... ({len(pbs) - 25} more)")

    return 0


# ----------------------------
# rebuild commands (CI contract)
# ----------------------------

def _resolve_db_path(global_db: str) -> str:
    return global_db or os.environ.get("MGC_DB") or DEFAULT_DB


def _resolve_artifacts_dir(default_out_dir: str) -> Path:
    root = (os.environ.get("MGC_ARTIFACTS_DIR") or "").strip()
    if not root:
        return Path(default_out_dir)
    base = Path(root) / "rebuild"
    leaf = Path(default_out_dir).name
    return base / leaf


def _determinism_check(builder) -> None:
    a = builder()
    b = builder()
    if stable_dumps(a) != stable_dumps(b):
        raise ValueError("Non-deterministic rebuild output detected")


def cmd_rebuild_ls(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    db = DBConn(Path(db_path))

    out: Dict[str, Any] = {"db": str(db.path), "tables": [], "counts": {}}
    with db.connect() as conn:
        out["tables"] = [
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name ASC").fetchall()
        ]
        for t in ("events", "marketing_posts", "playlist_items", "playlist_runs", "playlists", "tracks"):
            if _table_exists(conn, t):
                out["counts"][t] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"])

    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_rebuild_playlists(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.rebuild.playlists")
    db_path = _resolve_db_path(args.db)
    out_dir = _resolve_artifacts_dir(str(args.out_dir))

    if args.stamp:
        log.debug("stamp=%s", args.stamp)

    db = DBConn(Path(db_path))
    written: List[str] = []

    with db.connect() as conn:
        rows = _safe_select_latest_playlists_by_slug(conn)
        for r in rows:
            d = _row_to_dict(r)
            slug = str(d.get("slug") or d.get("name") or "playlist")
            out_path = Path(out_dir) / f"{_safe_slug(slug)}.json"

            def build_one() -> Any:
                return _build_playlist_json_for_row(conn, d)

            if args.determinism_check:
                _determinism_check(build_one)

            obj = build_one()
            if args.write:
                write_json_file(out_path, obj)

            written.append(str(out_path))

    if args.json:
        print(json.dumps({"written": written}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for pth in written:
            print(pth)
    return 0


def cmd_rebuild_tracks(args: argparse.Namespace) -> int:
    log = logging.getLogger("mgc.rebuild.tracks")
    db_path = _resolve_db_path(args.db)
    out_dir = _resolve_artifacts_dir(str(args.out_dir))
    out_path = out_dir / DEFAULT_TRACKS_EXPORT.name

    if args.stamp:
        log.debug("stamp=%s", args.stamp)

    db = DBConn(Path(db_path))
    with db.connect() as conn:
        if not _table_exists(conn, "tracks"):
            if args.json:
                print(json.dumps({"written": []}, indent=2, ensure_ascii=False, sort_keys=True))
            return 0

        cols = set(_columns(conn, "tracks"))
        q = "SELECT * FROM tracks "
        if "created_at" in cols and "id" in cols:
            q += "ORDER BY created_at ASC, id ASC"
        elif "id" in cols:
            q += "ORDER BY id ASC"

        rows = conn.execute(q).fetchall()

        def build_one() -> Any:
            return [_row_to_dict(r) for r in rows]

        if args.determinism_check:
            _determinism_check(build_one)

        data = build_one()
        if args.write:
            write_json_file(out_path, data)

    if args.json:
        print(json.dumps({"written": [str(out_path)]}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(str(out_path))
    return 0


def cmd_rebuild_verify_playlists(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    out_dir = _resolve_artifacts_dir(str(args.out_dir))

    db = DBConn(Path(db_path))
    with db.connect() as conn:
        diffs: List[Dict[str, Any]] = []
        rows = _safe_select_latest_playlists_by_slug(conn)
        for r in rows:
            d = _row_to_dict(r)
            slug = str(d.get("slug") or d.get("name") or "playlist")
            path = Path(out_dir) / f"{_safe_slug(slug)}.json"
            expected = _build_playlist_json_for_row(conn, d)

            if not path.exists():
                diffs.append({"path": str(path), "reason": "missing"})
                continue

            try:
                actual = read_json_file(path)
            except Exception:
                diffs.append({"path": str(path), "reason": "unreadable"})
                continue

            if stable_dumps(actual) != stable_dumps(expected):
                diffs.append({"path": str(path), "reason": "content_diff"})

    payload = {"diffs": diffs, "diff_count": len(diffs)}

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for d in diffs:
            print(f"{d['path']} {d['reason']}")

    if diffs and args.strict:
        return 2
    return 0


def cmd_rebuild_verify_tracks(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    out_dir = _resolve_artifacts_dir(str(args.out_dir))
    out_path = out_dir / DEFAULT_TRACKS_EXPORT.name

    diffs: List[Dict[str, Any]] = []
    db = DBConn(Path(db_path))

    with db.connect() as conn:
        if not _table_exists(conn, "tracks"):
            payload = {"diffs": [], "diff_count": 0}
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                print("tracks table missing")
            return 0

        if not out_path.exists():
            diffs.append({"path": str(out_path), "reason": "missing"})
        else:
            try:
                actual = read_json_file(out_path)
            except Exception:
                diffs.append({"path": str(out_path), "reason": "unreadable"})
                actual = None

            if actual is not None:
                cols = set(_columns(conn, "tracks"))
                q = "SELECT * FROM tracks "
                if "created_at" in cols and "id" in cols:
                    q += "ORDER BY created_at ASC, id ASC"
                elif "id" in cols:
                    q += "ORDER BY id ASC"
                rows = conn.execute(q).fetchall()
                expected = [_row_to_dict(r) for r in rows]
                if stable_dumps(actual) != stable_dumps(expected):
                    diffs.append({"path": str(out_path), "reason": "content_diff"})

    payload = {"diffs": diffs, "diff_count": len(diffs)}

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        for d in diffs:
            print(f"{d['path']} {d['reason']}")

    if diffs and args.strict:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Single source of truth parser builder.
    """
    p = argparse.ArgumentParser(prog="mgc", description="Music Generator Company CLI")

    # GLOBALS ONLY: do NOT repeat these on subparsers
    p.add_argument("--db", default=os.environ.get("MGC_DB", DEFAULT_DB))
    p.add_argument("--repo-root", default=os.environ.get("MGC_REPO_ROOT", "."), help="Repository root (CI may pass this before subcommands)")
    p.add_argument("--log-level", default=os.environ.get("MGC_LOG_LEVEL", "INFO"))
    p.add_argument("--log-file", default=os.environ.get("MGC_LOG_FILE"))
    p.add_argument("--log-console", action="store_true")
    p.add_argument("--json", action="store_true", help="Output JSON where supported")

    # CI harness compatibility (accept but don't require downstream usage)
    p.add_argument("--seed", type=int, default=int(os.environ.get("MGC_SEED", "1")), help="Random seed (CI compatibility)")
    p.add_argument("--no-resume", action="store_true", help="Disable resume behavior (CI compatibility)")

    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="Show health + recent activity snapshot")
    st.set_defaults(func=cmd_status)

    # providers
    prov = sub.add_parser("providers", help="Provider tools (list/check env configuration)")
    provs = prov.add_subparsers(dest="providers_cmd", required=True)

    prov_list = provs.add_parser("list", help="List supported providers and required env vars")
    prov_list.set_defaults(func=cmd_providers_list)

    prov_check = provs.add_parser("check", help="Validate env for selected provider (MGC_PROVIDER)")
    prov_check.set_defaults(func=cmd_providers_check)

    # db
    dbp = sub.add_parser("db", help="Database tools (migrations)")
    dbs = dbp.add_subparsers(dest="db_cmd", required=True)

    dbm = dbs.add_parser("migrate", help="Apply SQL migrations (scripts/migrations)")
    dbm.add_argument("--migrations-dir", default=None, help="Override migrations dir (default: <repo-root>/scripts/migrations)")
    dbm.add_argument("--now", default=None, help="Override applied_at timestamp (ISO; supports Z)")
    dbm.set_defaults(func=cmd_db_migrate)

    dbs_ = dbs.add_parser("status", help="Show applied migrations and billing table presence")
    dbs_.add_argument("--migrations-dir", default=None, help="Override migrations dir (default: <repo-root>/scripts/migrations)")
    dbs_.set_defaults(func=cmd_db_status)

    rebuild = sub.add_parser("rebuild", help="Rebuild derived artifacts deterministically (CI)")
    rs = rebuild.add_subparsers(dest="rebuild_cmd", required=True)

    rls = rs.add_parser("ls", help="Show rebuild/status info (CI expects this)")
    rls.set_defaults(func=cmd_rebuild_ls)

    rp = rs.add_parser("playlists", help="Rebuild canonical playlist JSON files (CI)")
    rp.add_argument("--out-dir", default=str(DEFAULT_PLAYLIST_DIR))
    rp.add_argument("--stamp", default=None)
    rp.add_argument("--determinism-check", action="store_true")
    rp.add_argument("--write", action="store_true")
    rp.set_defaults(func=cmd_rebuild_playlists)

    rt = rs.add_parser("tracks", help="Rebuild canonical tracks export (CI)")
    rt.add_argument("--out-dir", default=str(DEFAULT_TRACKS_DIR))
    rt.add_argument("--stamp", default=None)
    rt.add_argument("--determinism-check", action="store_true")
    rt.add_argument("--write", action="store_true")
    rt.set_defaults(func=cmd_rebuild_tracks)

    rv = rs.add_parser("verify", help="Verify rebuilt outputs match DB + files (CI expects this)")
    rvs = rv.add_subparsers(dest="verify_cmd", required=True)

    rvp = rvs.add_parser("playlists", help="Verify playlists outputs")
    rvp.add_argument("--out-dir", default=str(DEFAULT_PLAYLIST_DIR))
    rvp.add_argument("--stamp", default=None)
    rvp.add_argument("--strict", action="store_true")
    rvp.add_argument("rest", nargs=argparse.REMAINDER)
    rvp.set_defaults(func=cmd_rebuild_verify_playlists)

    rvt = rvs.add_parser("tracks", help="Verify tracks outputs")
    rvt.add_argument("--out-dir", default=str(DEFAULT_TRACKS_DIR))
    rvt.add_argument("--stamp", default=None)
    rvt.add_argument("--strict", action="store_true")
    rvt.add_argument("rest", nargs=argparse.REMAINDER)
    rvt.set_defaults(func=cmd_rebuild_verify_tracks)

    playlists = sub.add_parser("playlists", help="Inspect playlists")
    ps = playlists.add_subparsers(dest="playlists_cmd", required=True)

    pl = ps.add_parser("list", help="List playlists")
    pl.add_argument("--limit", type=int, default=20)
    pl.add_argument("--slug", default=None)
    pl.set_defaults(func=cmd_playlists_list)

    ph = ps.add_parser("history", help="Show playlist history for a slug (or id)")
    ph.add_argument("target")
    ph.add_argument("--limit", type=int, default=20)
    ph.set_defaults(func=cmd_playlists_history)

    pr = ps.add_parser("reveal", help="Print playlist JSON for a playlist id")
    pr.add_argument("id")
    pr.set_defaults(func=cmd_playlists_reveal)

    pe = ps.add_parser("export", help="Export recent playlist JSON files")
    pe.add_argument("--limit", type=int, default=1)
    pe.add_argument("--slug", default=None)
    pe.add_argument("--out-dir", default=str(DEFAULT_PLAYLIST_DIR))
    pe.set_defaults(func=cmd_playlists_export)

    tracks = sub.add_parser("tracks", help="Inspect track library")
    ts = tracks.add_subparsers(dest="tracks_cmd", required=True)

    tl = ts.add_parser("list", help="List tracks")
    tl.add_argument("--limit", type=int, default=20)
    tl.add_argument("--mood", default=None)
    tl.add_argument("--genre", default=None)
    tl.set_defaults(func=cmd_tracks_list)

    tshow = ts.add_parser("show", help="Show track details")
    tshow.add_argument("id")
    tshow.set_defaults(func=cmd_tracks_show)

    tstats = ts.add_parser("stats", help="Track library stats")
    tstats.set_defaults(func=cmd_tracks_stats)

    marketing = sub.add_parser("marketing", help="Marketing tools")
    ms = marketing.add_subparsers(dest="marketing_cmd", required=True)

    mposts = ms.add_parser("posts", help="Marketing posts")
    mps = mposts.add_subparsers(dest="posts_cmd", required=True)

    mpl = mps.add_parser("list", help="List marketing posts")
    mpl.add_argument("--limit", type=int, default=20)
    mpl.set_defaults(func=cmd_marketing_posts_list)

    # receipts (file-based)
    mreceipts = ms.add_parser("receipts", help="Publish receipts (file-based audit trail)")
    mrs = mreceipts.add_subparsers(dest="receipts_cmd", required=True)

    mrl = mrs.add_parser("list", help="List receipts from a JSONL file")
    mrl.add_argument("--path", default="artifacts/run/marketing/receipts.jsonl", help="Path to receipts.jsonl")
    mrl.add_argument("--limit", type=int, default=20, help="Number of receipts to show (newest last)")
    mrl.set_defaults(func=cmd_marketing_receipts_list)


    # Optional: agents (only if agents_cli is present)
    try:
        from mgc.agents_cli import register_agents_subcommand  # type: ignore
    except Exception:
        register_agents_subcommand = None  # type: ignore
    if register_agents_subcommand:
        register_agents_subcommand(sub)

    try:
        from mgc.drops_cli import register_drops_subcommand  # type: ignore
    except Exception:
        register_drops_subcommand = None  # type: ignore
    if register_drops_subcommand:
        register_drops_subcommand(sub)

    try:
        from mgc.analytics_cli import register_analytics_subcommand  # type: ignore
    except Exception:
        register_analytics_subcommand = None  # type: ignore
    if register_analytics_subcommand:
        register_analytics_subcommand(sub)

    # web (required by workflows)
    try:
        import mgc.web_cli as _web_cli  # type: ignore
    except Exception as e:
        raise SystemExit(f"[mgc.main] ERROR: failed to import mgc.web_cli: {e}") from e

    _web_registrar = None
    for _name in (
        "register_web_subcommand",
        "register_web_subparser",
        "register_web_cli",
        "register_web",
        "register_subcommand",
        "register",
    ):
        _maybe = getattr(_web_cli, _name, None)
        if callable(_maybe):
            _web_registrar = _maybe
            break

    if _web_registrar is None:
        _pub = [
            n for n in dir(_web_cli)
            if not n.startswith("_") and callable(getattr(_web_cli, n, None))
        ]
        raise SystemExit(
            "[mgc.main] ERROR: mgc.web_cli imported, but no web registrar function found. "
            "Expected one of: register_web_subcommand/register_web_subparser/register_web_cli/"
            "register_web/register_subcommand/register. "
            f"Callable exports: {_pub}"
        )
    _web_registrar(sub)

    # submission (deterministic submission ZIPs)
    try:
        register_submission_subcommand(sub)
    except Exception as e:
        print(f"[mgc.main] WARN: submission subcommand not registered: {e}")

    try:
        from mgc.run_cli import register_run_subcommand as _register_run_subcommand  # type: ignore
    except Exception as e:
        raise SystemExit(f"[mgc.main] ERROR: failed to import mgc.run_cli: {e}") from e
    _register_run_subcommand(sub)

    # Optional: billing (only if billing_cli is present)
    try:
        from mgc.billing_cli import register_billing_subcommand  # type: ignore
    except Exception:
        register_billing_subcommand = None  # type: ignore
    if register_billing_subcommand:
        register_billing_subcommand(sub)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()

    raw = list(argv) if argv is not None else sys.argv[1:]
    cooked = _hoist_global_flags(raw)

    args = parser.parse_args(cooked)

    # Surface CI compatibility flags to env for downstream code.
    if getattr(args, "seed", None) is not None:
        os.environ["MGC_SEED"] = str(args.seed)
    if getattr(args, "no_resume", False):
        os.environ["MGC_NO_RESUME"] = "1"

    _configure_logging(
        level=getattr(args, "log_level", "INFO"),
        log_file=getattr(args, "log_file", None),
        log_console=bool(getattr(args, "log_console", False)),
        json_mode=bool(getattr(args, "json", False)),
    )

    fn = getattr(args, "func", None) or getattr(args, "fn", None)
    if not callable(fn):
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
mgc.analytics_cli

Analytics CLI for the MGC SQLite schema:

Tables:
  tracks
  playlists
  playlist_items
  playlist_runs
  marketing_posts

Subcommands:
  analytics overview
  analytics tracks
  analytics playlists
  analytics runs
  analytics marketing
  analytics stability        # NEW: Jaccard similarity run-to-run by slug
  analytics reuse            # NEW: track reuse (by playlists + by runs)
  analytics duration         # NEW: target vs produced duration accuracy
  analytics export <dataset> # datasets include new ones

Wire into mgc.main:
  from mgc.analytics_cli import register_analytics_subcommand
  register_analytics_subcommand(root_subparsers)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------
# DB helpers
# ----------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        raise SystemExit(f"DB not found: {db_path}")
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    return con


def _fetch_all(con: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _fetch_one(con: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
    cur = con.execute(sql, params)
    row = cur.fetchone()
    cur.close()
    return row


# ----------------------------
# Formatting / printing
# ----------------------------

def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _print_kv(items: Iterable[Tuple[str, Any]]) -> None:
    items = list(items)
    if not items:
        print("(no data)")
        return
    w = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"{k:<{w}}  {_stringify(v)}")


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_width: int = 80) -> None:
    headers = list(headers)
    rows = [list(r) for r in rows]
    col_count = len(headers)

    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(col_count):
            widths[i] = max(widths[i], len(_stringify(r[i])))

    widths = [min(w, max_width) for w in widths]

    def clip(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        if w <= 1:
            return s[:w]
        return s[: w - 1] + "…"

    header_line = "  ".join(clip(headers[i], widths[i]).ljust(widths[i]) for i in range(col_count))
    sep_line = "  ".join("-" * widths[i] for i in range(col_count))
    print(header_line)
    print(sep_line)

    for r in rows:
        line = "  ".join(clip(_stringify(r[i]), widths[i]).ljust(widths[i]) for i in range(col_count))
        print(line)


def _rows_to_dicts(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d: Dict[str, Any] = {}
        for k in r.keys():
            d[k] = r[k]
        out.append(d)
    return out


# ----------------------------
# Core analytics (existing)
# ----------------------------

@dataclass(frozen=True)
class Overview:
    totals: Dict[str, Any]
    tracks_top_moods: List[Dict[str, Any]]
    tracks_top_genres: List[Dict[str, Any]]
    playlists_top_slugs: List[Dict[str, Any]]
    runs_top_slugs: List[Dict[str, Any]]
    marketing_top_platforms: List[Dict[str, Any]]


def _overview(con: sqlite3.Connection, top: int = 10) -> Overview:
    totals_row = _fetch_one(con, """
        SELECT
          (SELECT COUNT(*) FROM tracks)               AS tracks_total,
          (SELECT COUNT(*) FROM playlists)            AS playlists_total,
          (SELECT COUNT(*) FROM playlist_runs)        AS runs_total,
          (SELECT COUNT(*) FROM marketing_posts)      AS marketing_posts_total,

          (SELECT COUNT(*) FROM tracks
            WHERE datetime(created_at) >= datetime('now', '-7 days'))  AS tracks_last_7d,
          (SELECT COUNT(*) FROM tracks
            WHERE datetime(created_at) >= datetime('now', '-30 days')) AS tracks_last_30d,

          (SELECT COUNT(*) FROM playlists
            WHERE datetime(created_at) >= datetime('now', '-7 days'))  AS playlists_last_7d,
          (SELECT COUNT(*) FROM playlists
            WHERE datetime(created_at) >= datetime('now', '-30 days')) AS playlists_last_30d,

          (SELECT COUNT(*) FROM playlist_runs
            WHERE datetime(created_at) >= datetime('now', '-7 days'))  AS runs_last_7d,
          (SELECT COUNT(*) FROM playlist_runs
            WHERE datetime(created_at) >= datetime('now', '-30 days')) AS runs_last_30d,

          (SELECT ROUND(AVG(duration_sec), 2) FROM tracks)             AS avg_track_duration_sec,
          (SELECT ROUND(AVG(bpm), 1) FROM tracks)                      AS avg_track_bpm,
          (SELECT ROUND(AVG(track_count), 2) FROM playlists)           AS avg_playlist_track_count,
          (SELECT ROUND(AVG(total_duration_sec), 2) FROM playlists)    AS avg_playlist_total_duration_sec,
          (SELECT ROUND(AVG(target_minutes), 2) FROM playlists)        AS avg_playlist_target_minutes
    """)
    totals = dict(totals_row) if totals_row else {}

    moods = _rows_to_dicts(_fetch_all(con, """
        SELECT mood, COUNT(*) AS track_count
        FROM tracks
        GROUP BY mood
        ORDER BY track_count DESC, mood ASC
        LIMIT ?
    """, (top,)))

    genres = _rows_to_dicts(_fetch_all(con, """
        SELECT genre, COUNT(*) AS track_count
        FROM tracks
        GROUP BY genre
        ORDER BY track_count DESC, genre ASC
        LIMIT ?
    """, (top,)))

    slugs = _rows_to_dicts(_fetch_all(con, """
        SELECT slug, COUNT(*) AS playlist_count
        FROM playlists
        GROUP BY slug
        ORDER BY playlist_count DESC, slug ASC
        LIMIT ?
    """, (top,)))

    runs_by_slug = _rows_to_dicts(_fetch_all(con, """
        SELECT p.slug AS slug, COUNT(*) AS run_count
        FROM playlist_runs r
        JOIN playlists p ON p.id = r.playlist_id
        GROUP BY p.slug
        ORDER BY run_count DESC, p.slug ASC
        LIMIT ?
    """, (top,)))

    platforms = _rows_to_dicts(_fetch_all(con, """
        SELECT platform, COUNT(*) AS post_count
        FROM marketing_posts
        GROUP BY platform
        ORDER BY post_count DESC, platform ASC
        LIMIT ?
    """, (top,)))

    return Overview(
        totals=totals,
        tracks_top_moods=moods,
        tracks_top_genres=genres,
        playlists_top_slugs=slugs,
        runs_top_slugs=runs_by_slug,
        marketing_top_platforms=platforms,
    )


def _tracks_summary(con: sqlite3.Connection, top: int = 20, mood: Optional[str] = None,
                    genre: Optional[str] = None, status: Optional[str] = None) -> List[sqlite3.Row]:
    wh: List[str] = []
    params: List[Any] = []
    if mood:
        wh.append("mood = ?")
        params.append(mood)
    if genre:
        wh.append("genre = ?")
        params.append(genre)
    if status:
        wh.append("status = ?")
        params.append(status)

    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""
    sql = f"""
        SELECT
          id,
          created_at,
          title,
          mood,
          genre,
          bpm,
          duration_sec,
          status
        FROM tracks
        {where_sql}
        ORDER BY datetime(created_at) DESC
        LIMIT ?
    """
    params.append(top)
    return _fetch_all(con, sql, params)


def _tracks_aggregates(con: sqlite3.Connection, group_by: str, top: int = 20,
                       status: Optional[str] = None) -> List[sqlite3.Row]:
    if group_by not in ("mood", "genre", "status"):
        raise SystemExit("group_by must be one of: mood, genre, status")

    wh = ""
    params: List[Any] = []
    if status and group_by != "status":
        wh = "WHERE status = ?"
        params.append(status)

    sql = f"""
        SELECT
          {group_by} AS key,
          COUNT(*) AS track_count,
          ROUND(AVG(duration_sec), 2) AS avg_duration_sec,
          ROUND(AVG(bpm), 1) AS avg_bpm
        FROM tracks
        {wh}
        GROUP BY {group_by}
        ORDER BY track_count DESC, key ASC
        LIMIT ?
    """
    params.append(top)
    return _fetch_all(con, sql, params)


def _playlists_list(con: sqlite3.Connection, limit: int = 20, slug: Optional[str] = None,
                    context: Optional[str] = None) -> List[sqlite3.Row]:
    wh: List[str] = []
    params: List[Any] = []
    if slug:
        wh.append("slug = ?")
        params.append(slug)
    if context:
        wh.append("context = ?")
        params.append(context)

    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""

    return _fetch_all(con, f"""
        SELECT
          id,
          created_at,
          slug,
          name,
          context,
          mood,
          genre,
          bpm_min,
          bpm_max,
          target_minutes,
          track_count,
          total_duration_sec,
          json_path
        FROM playlists
        {where_sql}
        ORDER BY datetime(created_at) DESC
        LIMIT ?
    """, (*params, limit))


def _runs_list(con: sqlite3.Connection, limit: int = 50, slug: Optional[str] = None,
              playlist_id: Optional[str] = None) -> List[sqlite3.Row]:
    wh: List[str] = []
    params: List[Any] = []

    if playlist_id:
        wh.append("r.playlist_id = ?")
        params.append(playlist_id)
    if slug:
        wh.append("p.slug = ?")
        params.append(slug)

    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""

    return _fetch_all(con, f"""
        SELECT
          r.id,
          r.created_at,
          r.playlist_id,
          p.slug AS slug,
          p.name AS playlist_name,
          r.seed,
          json_array_length(r.track_ids_json) AS track_count_in_run,
          r.export_path,
          r.notes
        FROM playlist_runs r
        JOIN playlists p ON p.id = r.playlist_id
        {where_sql}
        ORDER BY datetime(r.created_at) DESC
        LIMIT ?
    """, (*params, limit))


def _marketing_list(con: sqlite3.Connection, limit: int = 50, platform: Optional[str] = None,
                    status: Optional[str] = None) -> List[sqlite3.Row]:
    wh: List[str] = []
    params: List[Any] = []
    if platform:
        wh.append("m.platform = ?")
        params.append(platform)
    if status:
        wh.append("m.status = ?")
        params.append(status)

    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""

    return _fetch_all(con, f"""
        SELECT
          m.id,
          m.created_at,
          m.platform,
          m.status,
          m.track_id,
          t.title AS track_title,
          t.mood AS track_mood,
          t.genre AS track_genre
        FROM marketing_posts m
        JOIN tracks t ON t.id = m.track_id
        {where_sql}
        ORDER BY datetime(m.created_at) DESC
        LIMIT ?
    """, (*params, limit))


# ----------------------------
# NEW: stability (Jaccard run-to-run by slug)
# ----------------------------

def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    inter = sa & sb
    return float(len(inter)) / float(len(union))


def _parse_track_ids_json(s: str) -> List[str]:
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def _stability_by_slug(con: sqlite3.Connection, per_slug_runs: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      (summary_rows, verbose_pairs)
    Summary is per slug (avg/min/max of consecutive-run Jaccard).
    Verbose contains row-per-comparison for debugging.
    """
    rows = _fetch_all(con, """
        SELECT
          p.slug AS slug,
          r.id   AS run_id,
          r.created_at AS created_at,
          r.track_ids_json AS track_ids_json
        FROM playlist_runs r
        JOIN playlists p ON p.id = r.playlist_id
        ORDER BY p.slug ASC, datetime(r.created_at) DESC
    """)

    # group by slug, keep most recent N runs
    by_slug: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        slug = str(r["slug"])
        by_slug.setdefault(slug, [])
        if len(by_slug[slug]) >= per_slug_runs:
            continue
        by_slug[slug].append(
            {
                "run_id": str(r["run_id"]),
                "created_at": str(r["created_at"]),
                "track_ids": _parse_track_ids_json(str(r["track_ids_json"])),
            }
        )

    summaries: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []

    for slug, runs in by_slug.items():
        # runs are in DESC order; compare consecutive pairs (newer vs next older)
        if len(runs) < 2:
            summaries.append(
                {"slug": slug, "comparisons": 0, "avg_jaccard": None, "min_jaccard": None, "max_jaccard": None}
            )
            continue

        sims: List[float] = []
        for i in range(len(runs) - 1):
            a = runs[i]
            b = runs[i + 1]
            sim = _jaccard(a["track_ids"], b["track_ids"])
            sims.append(sim)
            pairs.append(
                {
                    "slug": slug,
                    "newer_run_id": a["run_id"],
                    "older_run_id": b["run_id"],
                    "newer_created_at": a["created_at"],
                    "older_created_at": b["created_at"],
                    "jaccard": sim,
                    "newer_tracks": len(a["track_ids"]),
                    "older_tracks": len(b["track_ids"]),
                }
            )

        avg = sum(sims) / float(len(sims)) if sims else None
        summaries.append(
            {
                "slug": slug,
                "comparisons": len(sims),
                "avg_jaccard": avg,
                "min_jaccard": (min(sims) if sims else None),
                "max_jaccard": (max(sims) if sims else None),
            }
        )

    # sort: least stable first (avg_jaccard ascending, None last)
    def sort_key(d: Dict[str, Any]) -> Tuple[int, float]:
        v = d.get("avg_jaccard")
        if v is None:
            return (1, 1.0)
        return (0, float(v))

    summaries.sort(key=sort_key)
    return summaries, pairs


# ----------------------------
# NEW: reuse (tracks reused across playlists / runs)
# ----------------------------

def _reuse_by_playlists(con: sqlite3.Connection, top: int = 50) -> List[sqlite3.Row]:
    # How many playlists each track appears in (playlist_items)
    return _fetch_all(con, """
        SELECT
          pi.track_id AS track_id,
          COUNT(*) AS appearances,
          COUNT(DISTINCT pi.playlist_id) AS playlists,
          t.title AS title,
          t.mood AS mood,
          t.genre AS genre,
          t.status AS status
        FROM playlist_items pi
        JOIN tracks t ON t.id = pi.track_id
        GROUP BY pi.track_id
        ORDER BY appearances DESC, playlists DESC, t.title ASC
        LIMIT ?
    """, (top,))


def _reuse_by_runs(con: sqlite3.Connection, top: int = 50) -> List[Dict[str, Any]]:
    """
    Prefer SQL json_each if available. Fallback to python parse.
    """
    # Attempt JSON1 path
    try:
        rows = _fetch_all(con, """
            SELECT
              je.value AS track_id,
              COUNT(*) AS appearances
            FROM playlist_runs r,
                 json_each(r.track_ids_json) AS je
            GROUP BY je.value
            ORDER BY appearances DESC, track_id ASC
            LIMIT ?
        """, (top,))
        # join to tracks for title/mood/genre/status
        ids = [str(r["track_id"]) for r in rows]
        meta: Dict[str, Dict[str, Any]] = {}
        if ids:
            # chunk to avoid too many bind vars
            for i in range(0, len(ids), 900):
                chunk = ids[i : i + 900]
                q = ",".join(["?"] * len(chunk))
                mrows = _fetch_all(con, f"""
                    SELECT id, title, mood, genre, status
                    FROM tracks
                    WHERE id IN ({q})
                """, tuple(chunk))
                for mr in mrows:
                    meta[str(mr["id"])] = {
                        "title": mr["title"],
                        "mood": mr["mood"],
                        "genre": mr["genre"],
                        "status": mr["status"],
                    }

        out: List[Dict[str, Any]] = []
        for r in rows:
            tid = str(r["track_id"])
            out.append(
                {
                    "track_id": tid,
                    "appearances": int(r["appearances"]),
                    **meta.get(tid, {}),
                }
            )
        return out

    except sqlite3.OperationalError:
        pass

    # Fallback: python scan
    run_rows = _fetch_all(con, "SELECT track_ids_json FROM playlist_runs")
    counts: Dict[str, int] = {}
    for rr in run_rows:
        tids = _parse_track_ids_json(str(rr["track_ids_json"]))
        for tid in tids:
            counts[tid] = counts.get(tid, 0) + 1

    # fetch meta
    top_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    ids = [tid for tid, _ in top_items]
    meta: Dict[str, Dict[str, Any]] = {}
    if ids:
        for i in range(0, len(ids), 900):
            chunk = ids[i : i + 900]
            q = ",".join(["?"] * len(chunk))
            mrows = _fetch_all(con, f"""
                SELECT id, title, mood, genre, status
                FROM tracks
                WHERE id IN ({q})
            """, tuple(chunk))
            for mr in mrows:
                meta[str(mr["id"])] = {
                    "title": mr["title"],
                    "mood": mr["mood"],
                    "genre": mr["genre"],
                    "status": mr["status"],
                }

    out: List[Dict[str, Any]] = []
    for tid, n in top_items:
        out.append({"track_id": tid, "appearances": n, **meta.get(tid, {})})
    return out


# ----------------------------
# NEW: duration accuracy (target vs produced)
# ----------------------------

def _duration_accuracy(con: sqlite3.Connection, top: int = 50) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = _fetch_all(con, """
        SELECT
          slug,
          COUNT(*) AS n,
          AVG((total_duration_sec - (target_minutes * 60))) AS avg_delta_sec,
          AVG(ABS(total_duration_sec - (target_minutes * 60))) AS avg_abs_error_sec,
          MIN((total_duration_sec - (target_minutes * 60))) AS min_delta_sec,
          MAX((total_duration_sec - (target_minutes * 60))) AS max_delta_sec
        FROM playlists
        GROUP BY slug
        ORDER BY avg_abs_error_sec DESC
        LIMIT ?
    """, (top,))

    per_slug = []
    for r in rows:
        per_slug.append(
            {
                "slug": r["slug"],
                "n": int(r["n"]),
                "avg_delta_sec": float(r["avg_delta_sec"]) if r["avg_delta_sec"] is not None else None,
                "avg_abs_error_sec": float(r["avg_abs_error_sec"]) if r["avg_abs_error_sec"] is not None else None,
                "min_delta_sec": float(r["min_delta_sec"]) if r["min_delta_sec"] is not None else None,
                "max_delta_sec": float(r["max_delta_sec"]) if r["max_delta_sec"] is not None else None,
            }
        )

    overall_row = _fetch_one(con, """
        SELECT
          COUNT(*) AS n,
          AVG((total_duration_sec - (target_minutes * 60))) AS avg_delta_sec,
          AVG(ABS(total_duration_sec - (target_minutes * 60))) AS avg_abs_error_sec,
          MIN((total_duration_sec - (target_minutes * 60))) AS min_delta_sec,
          MAX((total_duration_sec - (target_minutes * 60))) AS max_delta_sec
        FROM playlists
    """)
    overall = dict(overall_row) if overall_row else {}
    # ensure numeric types for JSON friendliness
    if overall:
        for k in ("avg_delta_sec", "avg_abs_error_sec", "min_delta_sec", "max_delta_sec"):
            if overall.get(k) is not None:
                overall[k] = float(overall[k])
        if overall.get("n") is not None:
            overall["n"] = int(overall["n"])

    return overall, per_slug


# ----------------------------
# Export helpers
# ----------------------------

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _dataset(con: sqlite3.Connection, name: str, args: argparse.Namespace) -> Any:
    if name == "overview":
        o = _overview(con, top=getattr(args, "top", 10))
        return {
            "totals": o.totals,
            "top_moods": o.tracks_top_moods,
            "top_genres": o.tracks_top_genres,
            "top_playlists_by_slug": o.playlists_top_slugs,
            "top_runs_by_slug": o.runs_top_slugs,
            "top_marketing_platforms": o.marketing_top_platforms,
        }

    if name == "tracks":
        rows = _tracks_summary(
            con,
            top=getattr(args, "limit", 200),
            mood=getattr(args, "mood", None),
            genre=getattr(args, "genre", None),
            status=getattr(args, "status", None),
        )
        return _rows_to_dicts(rows)

    if name == "playlists":
        rows = _playlists_list(
            con,
            limit=getattr(args, "limit", 200),
            slug=getattr(args, "slug", None),
            context=getattr(args, "context", None),
        )
        return _rows_to_dicts(rows)

    if name == "runs":
        rows = _runs_list(
            con,
            limit=getattr(args, "limit", 500),
            slug=getattr(args, "slug", None),
            playlist_id=getattr(args, "playlist_id", None),
        )
        return _rows_to_dicts(rows)

    if name == "marketing":
        rows = _marketing_list(
            con,
            limit=getattr(args, "limit", 500),
            platform=getattr(args, "platform", None),
            status=getattr(args, "status", None),
        )
        return _rows_to_dicts(rows)

    if name == "stability":
        summaries, pairs = _stability_by_slug(con, per_slug_runs=getattr(args, "per_slug_runs", 10))
        if getattr(args, "verbose", False):
            return {"per_slug": summaries, "pairs": pairs}
        return {"per_slug": summaries}

    if name == "reuse_playlists":
        rows = _reuse_by_playlists(con, top=getattr(args, "top", 50))
        return _rows_to_dicts(rows)

    if name == "reuse_runs":
        return _reuse_by_runs(con, top=getattr(args, "top", 50))

    if name == "duration":
        overall, per_slug = _duration_accuracy(con, top=getattr(args, "top", 50))
        return {"overall": overall, "per_slug": per_slug}

    raise SystemExit(
        f"Unknown dataset: {name} (expected: overview, tracks, playlists, runs, marketing, stability, reuse_playlists, reuse_runs, duration)"
    )


# ----------------------------
# Command handlers
# ----------------------------

def analytics_overview_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        o = _overview(con, top=args.top)

        print("Totals")
        _print_kv([
            ("Tracks", o.totals.get("tracks_total")),
            ("Playlists", o.totals.get("playlists_total")),
            ("Runs", o.totals.get("runs_total")),
            ("Marketing posts", o.totals.get("marketing_posts_total")),
            ("Tracks last 7d", o.totals.get("tracks_last_7d")),
            ("Tracks last 30d", o.totals.get("tracks_last_30d")),
            ("Playlists last 7d", o.totals.get("playlists_last_7d")),
            ("Playlists last 30d", o.totals.get("playlists_last_30d")),
            ("Runs last 7d", o.totals.get("runs_last_7d")),
            ("Runs last 30d", o.totals.get("runs_last_30d")),
            ("Avg track duration (sec)", o.totals.get("avg_track_duration_sec")),
            ("Avg track BPM", o.totals.get("avg_track_bpm")),
            ("Avg playlist track count", o.totals.get("avg_playlist_track_count")),
            ("Avg playlist total duration (sec)", o.totals.get("avg_playlist_total_duration_sec")),
            ("Avg playlist target minutes", o.totals.get("avg_playlist_target_minutes")),
        ])

        if o.tracks_top_moods:
            print("\nTop moods")
            _print_table(["mood", "track_count"], [[d["mood"], d["track_count"]] for d in o.tracks_top_moods])

        if o.tracks_top_genres:
            print("\nTop genres")
            _print_table(["genre", "track_count"], [[d["genre"], d["track_count"]] for d in o.tracks_top_genres])

        if o.playlists_top_slugs:
            print("\nTop playlist slugs")
            _print_table(["slug", "playlist_count"], [[d["slug"], d["playlist_count"]] for d in o.playlists_top_slugs])

        if o.runs_top_slugs:
            print("\nTop run slugs")
            _print_table(["slug", "run_count"], [[d["slug"], d["run_count"]] for d in o.runs_top_slugs])

        if o.marketing_top_platforms:
            print("\nTop marketing platforms")
            _print_table(["platform", "post_count"], [[d["platform"], d["post_count"]] for d in o.marketing_top_platforms])

        return 0
    finally:
        con.close()


def analytics_tracks_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        if args.group_by:
            rows = _tracks_aggregates(con, group_by=args.group_by, top=args.top, status=args.status)
            _print_table(
                ["key", "track_count", "avg_duration_sec", "avg_bpm"],
                [[r["key"], r["track_count"], r["avg_duration_sec"], r["avg_bpm"]] for r in rows],
            )
            return 0

        rows = _tracks_summary(con, top=args.top, mood=args.mood, genre=args.genre, status=args.status)
        _print_table(
            ["id", "created_at", "title", "mood", "genre", "bpm", "duration_sec", "status"],
            [[r["id"], r["created_at"], r["title"], r["mood"], r["genre"], r["bpm"], r["duration_sec"], r["status"]] for r in rows],
            max_width=48,
        )
        return 0
    finally:
        con.close()


def analytics_playlists_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        rows = _playlists_list(con, limit=args.limit, slug=args.slug, context=args.context)
        _print_table(
            ["id", "created_at", "slug", "name", "context", "mood", "genre", "target_minutes", "track_count", "total_duration_sec"],
            [[
                r["id"], r["created_at"], r["slug"], r["name"], r["context"], r["mood"], r["genre"],
                r["target_minutes"], r["track_count"], r["total_duration_sec"]
            ] for r in rows],
            max_width=40,
        )
        return 0
    finally:
        con.close()


def analytics_runs_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        rows = _runs_list(con, limit=args.limit, slug=args.slug, playlist_id=args.playlist_id)
        _print_table(
            ["id", "created_at", "playlist_id", "slug", "seed", "track_count_in_run", "export_path", "notes"],
            [[
                r["id"], r["created_at"], r["playlist_id"], r["slug"], r["seed"],
                r["track_count_in_run"], r["export_path"], r["notes"]
            ] for r in rows],
            max_width=42,
        )
        return 0
    finally:
        con.close()


def analytics_marketing_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        rows = _marketing_list(con, limit=args.limit, platform=args.platform, status=args.status)
        _print_table(
            ["id", "created_at", "platform", "status", "track_id", "track_title", "track_mood", "track_genre"],
            [[
                r["id"], r["created_at"], r["platform"], r["status"], r["track_id"],
                r["track_title"], r["track_mood"], r["track_genre"]
            ] for r in rows],
            max_width=40,
        )
        return 0
    finally:
        con.close()


def analytics_stability_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        per_slug, pairs = _stability_by_slug(con, per_slug_runs=args.per_slug_runs)

        _print_table(
            ["slug", "comparisons", "avg_jaccard", "min_jaccard", "max_jaccard"],
            [[d["slug"], d["comparisons"], d["avg_jaccard"], d["min_jaccard"], d["max_jaccard"]] for d in per_slug],
            max_width=40,
        )

        if args.verbose:
            print("\nPairs (newer vs older)")
            _print_table(
                ["slug", "jaccard", "newer_created_at", "older_created_at", "newer_run_id", "older_run_id"],
                [[
                    p["slug"], p["jaccard"], p["newer_created_at"], p["older_created_at"],
                    p["newer_run_id"], p["older_run_id"]
                ] for p in pairs[: args.pairs_limit]],
                max_width=40,
            )
            if len(pairs) > args.pairs_limit:
                print(f"\n… ({len(pairs) - args.pairs_limit} more pairs not shown; increase --pairs-limit)")
        return 0
    finally:
        con.close()


def analytics_reuse_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        if args.source == "playlists":
            rows = _reuse_by_playlists(con, top=args.top)
            _print_table(
                ["track_id", "appearances", "playlists", "title", "mood", "genre", "status"],
                [[r["track_id"], r["appearances"], r["playlists"], r["title"], r["mood"], r["genre"], r["status"]] for r in rows],
                max_width=44,
            )
            return 0

        if args.source == "runs":
            rows = _reuse_by_runs(con, top=args.top)
            _print_table(
                ["track_id", "appearances", "title", "mood", "genre", "status"],
                [[r.get("track_id"), r.get("appearances"), r.get("title"), r.get("mood"), r.get("genre"), r.get("status")] for r in rows],
                max_width=44,
            )
            return 0

        raise SystemExit("reuse --source must be playlists or runs")
    finally:
        con.close()


def analytics_duration_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        overall, per_slug = _duration_accuracy(con, top=args.top)

        print("Overall")
        _print_kv([
            ("n_playlists", overall.get("n")),
            ("avg_delta_sec", overall.get("avg_delta_sec")),
            ("avg_abs_error_sec", overall.get("avg_abs_error_sec")),
            ("min_delta_sec", overall.get("min_delta_sec")),
            ("max_delta_sec", overall.get("max_delta_sec")),
        ])

        print("\nWorst slugs by avg_abs_error_sec")
        _print_table(
            ["slug", "n", "avg_abs_error_sec", "avg_delta_sec", "min_delta_sec", "max_delta_sec"],
            [[d["slug"], d["n"], d["avg_abs_error_sec"], d["avg_delta_sec"], d["min_delta_sec"], d["max_delta_sec"]] for d in per_slug],
            max_width=40,
        )
        return 0
    finally:
        con.close()


def analytics_export_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        payload = _dataset(con, args.dataset, args)

        out_path = Path(args.out) if args.out else None
        if out_path is None:
            out_dir = Path("data/analytics")
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = "json" if args.format == "json" else "csv"
            out_path = out_dir / f"{args.dataset}.{ext}"

        if args.format == "json":
            _write_json(out_path, payload)
        elif args.format == "csv":
            if isinstance(payload, list):
                _write_csv(out_path, payload)
            elif isinstance(payload, dict):
                # CSV wants list[dict]; flatten a dict payload into one row
                _write_csv(out_path, [payload])
            else:
                raise SystemExit("CSV export expects dict or list[dict].")
        else:
            raise SystemExit("format must be json or csv")

        print(str(out_path))
        return 0
    finally:
        con.close()


# ----------------------------
# Argparse wiring
# ----------------------------

def register_analytics_subcommand(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("analytics", help="Analytics and reporting")
    aps = ap.add_subparsers(dest="analytics_cmd", required=True)

    def add_db_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")

    # overview
    ov = aps.add_parser("overview", help="High-level system metrics")
    add_db_arg(ov)
    ov.add_argument("--top", type=int, default=10, help="Top-N for breakdown tables")
    ov.set_defaults(func=analytics_overview_cmd)

    # tracks
    tr = aps.add_parser("tracks", help="Track analytics (list or aggregates)")
    add_db_arg(tr)
    tr.add_argument("--top", type=int, default=20, help="Row limit / Top-N")
    tr.add_argument("--mood", default=None)
    tr.add_argument("--genre", default=None)
    tr.add_argument("--status", default=None)
    tr.add_argument("--group-by", choices=["mood", "genre", "status"], default=None,
                    help="Aggregate by a dimension instead of listing tracks")
    tr.set_defaults(func=analytics_tracks_cmd)

    # playlists
    pl = aps.add_parser("playlists", help="Playlist analytics")
    add_db_arg(pl)
    pl.add_argument("--limit", type=int, default=20)
    pl.add_argument("--slug", default=None)
    pl.add_argument("--context", default=None)
    pl.set_defaults(func=analytics_playlists_cmd)

    # runs
    rn = aps.add_parser("runs", help="Playlist run analytics")
    add_db_arg(rn)
    rn.add_argument("--limit", type=int, default=50)
    rn.add_argument("--slug", default=None, help="Filter by playlist slug")
    rn.add_argument("--playlist-id", default=None, help="Filter by playlist id")
    rn.set_defaults(func=analytics_runs_cmd)

    # marketing
    mk = aps.add_parser("marketing", help="Marketing post analytics")
    add_db_arg(mk)
    mk.add_argument("--limit", type=int, default=50)
    mk.add_argument("--platform", default=None)
    mk.add_argument("--status", default=None)
    mk.set_defaults(func=analytics_marketing_cmd)

    # stability (NEW)
    st = aps.add_parser("stability", help="Run-to-run stability by slug (Jaccard similarity)")
    add_db_arg(st)
    st.add_argument("--per-slug-runs", type=int, default=10, help="How many most-recent runs per slug to include")
    st.add_argument("--verbose", action="store_true", help="Show per-pair comparisons")
    st.add_argument("--pairs-limit", type=int, default=50, help="Max pairs to show in verbose mode")
    st.set_defaults(func=analytics_stability_cmd)

    # reuse (NEW)
    ru = aps.add_parser("reuse", help="Track reuse across playlists or runs")
    add_db_arg(ru)
    ru.add_argument("--source", choices=["playlists", "runs"], default="playlists",
                    help="Reuse source: playlist_items or playlist_runs")
    ru.add_argument("--top", type=int, default=50)
    ru.set_defaults(func=analytics_reuse_cmd)

    # duration (NEW)
    du = aps.add_parser("duration", help="Target vs produced duration accuracy")
    add_db_arg(du)
    du.add_argument("--top", type=int, default=50, help="Top-N slugs by worst avg abs error")
    du.set_defaults(func=analytics_duration_cmd)

    # export
    ex = aps.add_parser("export", help="Export analytics datasets (json/csv)")
    add_db_arg(ex)
    ex.add_argument(
        "dataset",
        choices=[
            "overview",
            "tracks",
            "playlists",
            "runs",
            "marketing",
            "stability",
            "reuse_playlists",
            "reuse_runs",
            "duration",
        ],
    )
    ex.add_argument("--format", choices=["json", "csv"], default="json")
    ex.add_argument("--out", default=None, help="Output path (default: data/analytics/<dataset>.<ext>)")

    # dataset-specific filters used by _dataset()
    ex.add_argument("--top", type=int, default=10, help="Top-N for overview / reuse / duration")
    ex.add_argument("--limit", type=int, default=200, help="Row limit for list datasets")
    ex.add_argument("--mood", default=None)
    ex.add_argument("--genre", default=None)
    ex.add_argument("--status", default=None)
    ex.add_argument("--slug", default=None)
    ex.add_argument("--context", default=None)
    ex.add_argument("--playlist-id", default=None)
    ex.add_argument("--platform", default=None)
    ex.add_argument("--per-slug-runs", type=int, default=10, help="For stability export")
    ex.add_argument("--verbose", action="store_true", help="For stability export include pair rows")

    ex.set_defaults(func=analytics_export_cmd)


# ----------------------------
# Optional: standalone entrypoint
# ----------------------------

def _standalone_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mgc-analytics")
    sub = parser.add_subparsers(dest="cmd", required=True)
    register_analytics_subcommand(sub)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(_standalone_main())

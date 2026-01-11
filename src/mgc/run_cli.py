from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_stamp_default() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # rows: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def _try_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, (bytes, bytearray)):
        try:
            return json.loads(s.decode("utf-8"))
        except Exception:
            return None
    if isinstance(s, str):
        ss = s.strip()
        if not ss:
            return None
        try:
            return json.loads(ss)
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class EvidencePaths:
    root: Path

    @property
    def track_json(self) -> Path:
        return self.root / "track.json"

    @property
    def playlist_json(self) -> Path:
        return self.root / "playlist.json"

    @property
    def posts_dir(self) -> Path:
        return self.root / "marketing_posts"

    @property
    def summary_json(self) -> Path:
        return self.root / "run.json"


# ----------------------------
# run pipeline (generation → playlist → marketing → evidence)
# ----------------------------

def _insert_playlist_and_items(
    conn: sqlite3.Connection,
    *,
    playlist_id: str,
    created_at: str,
    slug: str,
    name: str,
    context: str,
    mood: Optional[str],
    genre: Optional[str],
    target_minutes: int,
    seed: int,
    total_duration_sec: int,
    track_ids: List[str],
    json_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO playlists (
          id, created_at, slug, name, context, mood, genre,
          target_minutes, seed, track_count, total_duration_sec, json_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            playlist_id,
            created_at,
            slug,
            name,
            context,
            mood,
            genre,
            int(target_minutes),
            int(seed),
            int(len(track_ids)),
            int(total_duration_sec),
            json_path,
        ),
    )

    for pos, tid in enumerate(track_ids, start=1):
        conn.execute(
            "INSERT INTO playlist_items (playlist_id, track_id, position) VALUES (?, ?, ?)",
            (playlist_id, tid, int(pos)),
        )


def cmd_run_daily(args: argparse.Namespace) -> int:
    # Local imports to keep CLI import-time light
    from mgc.storage import StoragePaths
    from mgc.db import DB
    from mgc.agents.music_agent import MusicAgent
    from mgc.agents.marketing_agent import MarketingAgent
    from mgc.playlist import build_playlist

    stamp = (args.stamp or _stable_stamp_default()).strip()
    db_path = Path(args.db)
    data_dir = Path(args.data_dir)
    artifacts_root = Path(args.artifacts_dir) / "runs" / stamp

    evidence = EvidencePaths(artifacts_root)
    _ensure_dir(evidence.root)
    _ensure_dir(evidence.posts_dir)

    storage = StoragePaths(data_dir=data_dir)
    storage.ensure()

    db = DB(db_path)
    db.init()

    # 1) Generate + store a track
    music = MusicAgent(storage=storage)
    track_art = music.run_daily()
    track_row = music.to_db_row(track_art)
    db.insert_track(track_row)
    _write_json(evidence.track_json, track_row)

    # 2) Build + store a playlist
    playlist_obj = build_playlist(
        db_path=db_path,
        context=args.context,
        slug=f"{args.context}_radio",
        name=f"{args.context.title()} Radio",
        target_minutes=int(args.target_minutes),
        seed=int(args.seed),
        lookback_playlists=int(args.lookback_playlists),
    )

    playlist_id = str(uuid.uuid4())
    created_at = _now_iso()
    playlist_slug = f"{args.context}_radio"

    playlist_out_dir = data_dir / "playlists"
    _ensure_dir(playlist_out_dir)
    playlist_json_path = playlist_out_dir / f"{playlist_slug}.json"

    _write_json(playlist_json_path, playlist_obj)
    _write_json(evidence.playlist_json, playlist_obj)

    track_ids = [it["track_id"] for it in playlist_obj.get("items", []) if it.get("track_id")]
    total_dur = int(playlist_obj.get("stats", {}).get("total_duration_sec", 0) or 0)

    with db.connect() as conn:
        _insert_playlist_and_items(
            conn,
            playlist_id=playlist_id,
            created_at=created_at,
            slug=playlist_slug,
            name=str(playlist_obj.get("name") or f"{args.context.title()} Radio"),
            context=args.context,
            mood=str(playlist_obj.get("filters", {}).get("mood") or "") or None,
            genre=str(playlist_obj.get("filters", {}).get("genre") or "") or None,
            target_minutes=int(args.target_minutes),
            seed=int(args.seed),
            total_duration_sec=total_dur,
            track_ids=track_ids,
            json_path=str(playlist_json_path),
        )
        conn.commit()

    # 3) Marketing drafts + DB rows
    marketing = MarketingAgent(storage=storage)
    posts = marketing.plan_posts(track_art)
    for p in posts:
        db.insert_post(p)

    # Mirror drafts into evidence bundle (best-effort)
    try:
        posts_dir = Path(storage.posts_dir)
        for draft in posts_dir.glob(f"{track_art.track_id}_*.json"):
            dst = evidence.posts_dir / draft.name
            dst.write_text(draft.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    run_summary: Dict[str, Any] = {
        "stamp": stamp,
        "created_at": created_at,
        "db": str(db_path),
        "data_dir": str(data_dir),
        "provider": os.getenv("MGC_PROVIDER", "stub"),
        "context": args.context,
        "track": {
            "id": track_art.track_id,
            "title": track_art.title,
            "full_path": track_art.full_path,
            "preview_path": track_art.preview_path,
        },
        "playlist": {
            "id": playlist_id,
            "slug": playlist_slug,
            "json_path": str(playlist_json_path),
            "track_count": len(track_ids),
            "total_duration_sec": total_dur,
        },
        "marketing_posts": {
            "count": len(posts),
            "platforms": sorted({p.get("platform") for p in posts if p.get("platform")}),
        },
        "evidence_dir": str(evidence.root),
    }
    _write_json(evidence.summary_json, run_summary)

    if args.json:
        print(json.dumps(run_summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(str(evidence.summary_json))

    return 0


# ----------------------------
# web player (static build + serve)
# ----------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MGC Player</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #444; margin-bottom: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px 14px; margin: 12px 0; }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
    .title {{ font-weight: 700; }}
    .small {{ color: #555; font-size: 0.95rem; }}
    audio {{ width: 100%; margin-top: 8px; }}
    code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MGC Player</h1>
    <div class="meta">
      Built at <code id="builtAt"></code> • Playlist: <code id="plistName"></code>
    </div>

    <div id="stats" class="small"></div>
    <div id="tracks"></div>

    <div class="meta small">
      Tip: serve the repo root so <code>/data/...</code> paths resolve.
      Example: <code>python -m mgc.main web serve</code>
    </div>
  </div>

<script>
  const PLAYLIST = __PLAYLIST_JSON__;
  const BUILT_AT = __BUILT_AT__;

  document.getElementById("builtAt").textContent = BUILT_AT;
  document.getElementById("plistName").textContent = PLAYLIST.name || "playlist";

  const s = PLAYLIST.stats || {{}};
  const statsText = [
    s.track_count != null ? `tracks=${{s.track_count}}` : null,
    s.duration_minutes != null ? `minutes=${{s.duration_minutes}}` : null,
    s.avg_bpm != null ? `avg_bpm=${{s.avg_bpm}}` : null,
  ].filter(Boolean).join(" • ");
  document.getElementById("stats").textContent = statsText;

  const wrap = document.getElementById("tracks");
  const items = PLAYLIST.items || [];
  if (!items.length) {{
    wrap.innerHTML = "<div class='card'>No tracks found in playlist JSON.</div>";
  }} else {{
    for (const [idx, t] of items.entries()) {{
      const card = document.createElement("div");
      card.className = "card";

      const title = document.createElement("div");
      title.className = "row";
      title.innerHTML = `
        <div>
          <span class="title">${{idx + 1}}. ${{t.title || t.track_id}}</span>
          <div class="small">${{t.mood || ""}} • ${{t.genre || ""}} • bpm=${{t.bpm ?? "?"}} • ${{Math.round((t.duration_sec || 0))}}s</div>
        </div>
        <div class="small"><code>${{t.track_id}}</code></div>
      `;

      const audio = document.createElement("audio");
      audio.controls = true;
      const src = document.createElement("source");
      src.src = "/" + (t.preview_path || t.full_path || "").replace(/^\\/+/, "");
      src.type = "audio/mpeg";
      audio.appendChild(src);

      const links = document.createElement("div");
      links.className = "small";
      const full = "/" + (t.full_path || "").replace(/^\\/+/, "");
      links.innerHTML = full ? `Full: <a href="${{full}}">${{full}}</a>` : "";

      card.appendChild(title);
      card.appendChild(audio);
      card.appendChild(links);
      wrap.appendChild(card);
    }}
  }}
</script>
</body>
</html>
"""


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(args.playlist).resolve()
    if not playlist_path.exists():
        raise SystemExit(f"playlist JSON not found: {playlist_path}")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    out_path = out_dir / "index.html"

    playlist_obj = _read_json(playlist_path)
    built_at = _now_iso()

    html = _HTML_TEMPLATE
    html = html.replace("__PLAYLIST_JSON__", json.dumps(playlist_obj, ensure_ascii=False))
    html = html.replace("__BUILT_AT__", json.dumps(built_at))

    out_path.write_text(html, encoding="utf-8")

    if args.copy_playlist:
        dst = out_dir / "playlist.json"
        shutil.copyfile(playlist_path, dst)

    if args.json:
        print(json.dumps({"written": str(out_path), "playlist": str(playlist_path), "built_at": built_at}, indent=2))
    else:
        print(str(out_path))

    return 0


def cmd_web_serve(args: argparse.Namespace) -> int:
    serve_dir = Path(args.dir).resolve()
    if not serve_dir.exists():
        raise SystemExit(f"serve dir not found: {serve_dir}")

    host = args.host
    port = int(args.port)

    os.chdir(str(serve_dir))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            if args.quiet:
                return
            super().log_message(fmt, *a)

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/artifacts/player/index.html"
    print(f"[web] serving: {serve_dir}")
    print(f"[web] open:   {url}")
    print("[web] stop with Ctrl+C")

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

    return 0


# ----------------------------
# publish simulation + receipts
# ----------------------------

def _simulated_permalink(platform: str, post_id: str) -> str:
    base = "https://example.invalid"
    platform = (platform or "unknown").strip().lower()
    return f"{base}/{platform}/{post_id}"


def _load_marketing_posts(
    conn: sqlite3.Connection,
    *,
    limit: int,
    only_status: Optional[str],
) -> List[Dict[str, Any]]:
    if not _table_exists(conn, "marketing_posts"):
        return []

    cols = set(_columns(conn, "marketing_posts"))

    # Support both schemas:
    # A) MVP: (id, created_at, track_id, platform, payload_json, status)
    # B) CI fixture: (id, created_at, status, title, body, platform, track_id, payload_json, metadata_json)
    want = ["id", "created_at", "track_id", "platform", "status"]
    if "payload_json" in cols:
        want.append("payload_json")
    if "metadata_json" in cols:
        want.append("metadata_json")
    if "title" in cols:
        want.append("title")
    if "body" in cols:
        want.append("body")

    select_cols = [c for c in want if c in cols]
    if not select_cols:
        select_cols = ["id"]  # last resort

    where = []
    params: List[Any] = []
    if only_status and "status" in cols:
        where.append("status = ?")
        params.append(only_status)

    q = f"SELECT {', '.join(select_cols)} FROM marketing_posts "
    if where:
        q += "WHERE " + " AND ".join(where) + " "
    if "created_at" in cols:
        q += "ORDER BY datetime(created_at) ASC "
    q += "LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(q, params).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        # parse payload/metadata if present
        if "payload_json" in d:
            d["payload"] = _try_json_loads(d.get("payload_json"))
        if "metadata_json" in d:
            d["metadata"] = _try_json_loads(d.get("metadata_json"))
        out.append(d)
    return out


def _update_marketing_post_status(
    conn: sqlite3.Connection,
    *,
    post_id: str,
    new_status: str,
    published_at: str,
) -> None:
    cols = set(_columns(conn, "marketing_posts"))
    if "status" not in cols:
        return

    # We avoid schema migrations; store publish time in metadata_json if available, else payload_json.
    meta_target = "metadata_json" if "metadata_json" in cols else ("payload_json" if "payload_json" in cols else None)

    if meta_target:
        row = conn.execute(
            f"SELECT {meta_target} FROM marketing_posts WHERE id = ? LIMIT 1",
            (post_id,),
        ).fetchone()
        cur = _try_json_loads(row[0]) if row is not None else None
        if not isinstance(cur, dict):
            cur = {}
        cur["published_at"] = published_at
        cur["published_simulated"] = True
        blob = json.dumps(cur, ensure_ascii=False, sort_keys=True)

        conn.execute(
            f"UPDATE marketing_posts SET status = ?, {meta_target} = ? WHERE id = ?",
            (new_status, blob, post_id),
        )
    else:
        conn.execute(
            "UPDATE marketing_posts SET status = ? WHERE id = ?",
            (new_status, post_id),
        )


def cmd_publish_marketing(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    stamp = (args.stamp or _stable_stamp_default()).strip()
    receipts_dir = Path(args.artifacts_dir) / "receipts" / stamp / "marketing"
    _ensure_dir(receipts_dir)

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    published_at = _now_iso()
    only_status = "planned" if args.only_planned else None

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        posts = _load_marketing_posts(conn, limit=int(args.limit), only_status=only_status)

        results: List[Dict[str, Any]] = []

        for p in posts:
            post_id = str(p.get("id", ""))
            platform = str(p.get("platform", "unknown"))
            track_id = str(p.get("track_id", ""))

            payload = p.get("payload")
            if not isinstance(payload, dict):
                payload = {}

            caption = payload.get("caption") or p.get("body") or ""
            hashtags = payload.get("hashtags") or []
            preview_path = payload.get("preview_path") or ""

            receipt = {
                "post_id": post_id,
                "platform": platform,
                "track_id": track_id,
                "status_before": p.get("status"),
                "status_after": "published" if args.commit else p.get("status"),
                "published_at": published_at,
                "simulated": True,
                "committed": bool(args.commit),
                "permalink": _simulated_permalink(platform, post_id),
                "caption": caption,
                "hashtags": hashtags,
                "preview_path": preview_path,
            }

            out = receipts_dir / platform.lower() / f"{post_id}.json"
            _write_json(out, receipt)

            if args.commit:
                _update_marketing_post_status(conn, post_id=post_id, new_status="published", published_at=published_at)

            results.append(
                {
                    "post_id": post_id,
                    "platform": platform,
                    "receipt_path": str(out),
                    "permalink": receipt["permalink"],
                    "committed": bool(args.commit),
                }
            )

        if args.commit:
            conn.commit()

    summary = {
        "db": str(db_path),
        "stamp": stamp,
        "published_at": published_at,
        "count": len(posts),
        "only_planned": bool(args.only_planned),
        "commit": bool(args.commit),
        "receipts_dir": str(receipts_dir),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"[publish marketing] receipts_dir={receipts_dir} count={len(posts)} commit={bool(args.commit)}")

    return 0


# ----------------------------
# registration
# ----------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    # ---- run ----
    run = subparsers.add_parser("run", help="Run the autonomous pipeline (generation → storage → promotion)")
    rs = run.add_subparsers(dest="run_cmd", required=True)

    daily = rs.add_parser("daily", help="Run a daily music drop pipeline")
    daily.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"))
    daily.add_argument("--data-dir", default="data", help="Where tracks/previews/posts/playlists are written")
    daily.add_argument("--artifacts-dir", default="artifacts", help="Where evidence bundles are written")
    daily.add_argument("--stamp", default=None, help="Run stamp (default: UTC date YYYY-MM-DD)")
    daily.add_argument("--context", default="focus", choices=["focus", "workout", "sleep"])
    daily.add_argument("--target-minutes", type=int, default=20)
    daily.add_argument("--seed", type=int, default=1)
    daily.add_argument("--lookback-playlists", type=int, default=3)
    daily.add_argument("--json", action="store_true", help="Print JSON run summary")
    daily.set_defaults(func=cmd_run_daily)

    weekly = rs.add_parser("weekly", help="Alias for daily (schedule weekly drops too)")
    weekly.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"))
    weekly.add_argument("--data-dir", default="data")
    weekly.add_argument("--artifacts-dir", default="artifacts")
    weekly.add_argument("--stamp", default=None)
    weekly.add_argument("--context", default="focus", choices=["focus", "workout", "sleep"])
    weekly.add_argument("--target-minutes", type=int, default=20)
    weekly.add_argument("--seed", type=int, default=1)
    weekly.add_argument("--lookback-playlists", type=int, default=3)
    weekly.add_argument("--json", action="store_true")
    weekly.set_defaults(func=cmd_run_daily)

    # ---- web ----
    web = subparsers.add_parser("web", help="Build and serve a simple static web player")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    wb = ws.add_parser("build", help="Build static player HTML into artifacts/player/")
    wb.add_argument("--playlist", default="data/playlists/focus_radio.json", help="Playlist JSON path")
    wb.add_argument("--out-dir", default="artifacts/player", help="Output directory for the static site")
    wb.add_argument("--copy-playlist", action="store_true", help="Copy playlist.json next to index.html")
    wb.add_argument("--json", action="store_true", help="Print JSON output")
    wb.set_defaults(func=cmd_web_build)

    wsrv = ws.add_parser("serve", help="Serve the repo so the player can load /data audio paths")
    wsrv.add_argument("--dir", default=".", help="Directory to serve (default: repo root)")
    wsrv.add_argument("--host", default="127.0.0.1")
    wsrv.add_argument("--port", default=8000, type=int)
    wsrv.add_argument("--quiet", action="store_true", help="Suppress request logs")
    wsrv.set_defaults(func=cmd_web_serve)

    # ---- publish ----
    pub = subparsers.add_parser("publish", help="Publish simulation steps (with receipts)")
    ps = pub.add_subparsers(dest="publish_cmd", required=True)

    pm = ps.add_parser("marketing", help="Simulate publishing marketing posts + write receipts")
    pm.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"))
    pm.add_argument("--artifacts-dir", default="artifacts")
    pm.add_argument("--stamp", default=None, help="Receipt stamp (default: UTC date YYYY-MM-DD)")
    pm.add_argument("--limit", type=int, default=50)
    pm.add_argument("--only-planned", action="store_true", help="Only publish posts with status=planned")
    pm.add_argument("--commit", action="store_true", help="Update DB status to published (still simulated)")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=cmd_publish_marketing)

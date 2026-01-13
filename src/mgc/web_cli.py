#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Web player build/serve helpers.

Key behaviors:
- Builds a self-contained static site under: <out-dir>/<slug>/
  (default: data/web/<slug>/)
- Embeds the playlist JSON directly into index.html as:
    const playlist = {...};
  with each track containing:
    full_path = "tracks/<id>.<ext>"
- Validates audio with ffprobe and skips bad files (logs a warning).
- Optional MP3 conversion with ffmpeg:
    --prefer-mp3 converts valid input audio to mp3 and uses mp3 in the build.
- Optional --strip-paths rewrites track paths to relative files (default behavior).

This module is intentionally self-contained and CI-friendly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG = logging.getLogger("mgc.web")


# ----------------------------
# Utilities
# ----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _read_json(p: Path) -> Any:
    return json.loads(_read_text(p))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _safe_slug(s: str) -> str:
    s2 = (s or "").strip().replace(" ", "_")
    return s2 if s2 else "site"


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: List[str], *, check: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, check=check)


def _ffprobe_ok(path: Path) -> bool:
    """
    Return True if ffprobe can parse the audio file.
    """
    if not path.exists() or not path.is_file():
        return False
    ffprobe = _which("ffprobe")
    if not ffprobe:
        # If ffprobe isn't available, assume "ok" (but warn once).
        LOG.warning("ffprobe not found on PATH; skipping audio validation for: %s", path)
        return True

    cp = _run(
        [
            ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=False,
        capture=False,
    )
    return cp.returncode == 0


def _convert_to_mp3(src: Path, dst: Path) -> bool:
    """
    Convert src audio to mp3 via ffmpeg. Returns True on success.
    """
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        LOG.error("ffmpeg not found on PATH; cannot convert to mp3: %s", src)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Use a conservative mp3 encode that plays everywhere.
    # - 44.1kHz, 2ch, VBR-ish quality 2 (good), but keep it stable.
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ac",
        "2",
        "-ar",
        "44100",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(dst),
    ]
    cp = _run(cmd, check=False, capture=False)
    return cp.returncode == 0 and dst.exists() and dst.stat().st_size > 0


# ----------------------------
# Playlist handling
# ----------------------------

@dataclass(frozen=True)
class TrackIn:
    id: str
    title: str
    full_path: str  # may be absolute, relative, or already web-relative


def _coerce_track(obj: Dict[str, Any]) -> Optional[TrackIn]:
    tid = str(obj.get("id") or obj.get("track_id") or obj.get("slug") or "").strip()
    if not tid:
        return None

    title = str(obj.get("title") or obj.get("name") or tid).strip()

    # Prefer the key we already use in the web player.
    fp = obj.get("full_path") or obj.get("path") or obj.get("file") or obj.get("url") or obj.get("src") or ""
    fp = str(fp).strip()
    if not fp:
        return None

    return TrackIn(id=tid, title=title, full_path=fp)


def _load_playlist(playlist_path: Path) -> Dict[str, Any]:
    obj = _read_json(playlist_path)
    if not isinstance(obj, dict):
        raise ValueError(f"Playlist JSON must be an object: {playlist_path}")
    if "tracks" not in obj or not isinstance(obj["tracks"], list):
        raise ValueError(f"Playlist JSON must contain list field 'tracks': {playlist_path}")
    return obj


def _resolve_track_source(playlist_file: Path, track_full_path: str) -> Path:
    """
    Resolve a track path from playlist JSON to a local filesystem path.

    - If it's an absolute path, keep it.
    - If it's relative, interpret relative to the playlist file directory.
    """
    p = Path(track_full_path)
    if p.is_absolute():
        return p
    # common case: playlist stored under data/playlists and paths point into data/tracks
    return (playlist_file.parent / p).resolve()


# ----------------------------
# Static site template
# ----------------------------

def _render_index_html(*, playlist_obj: Dict[str, Any], title: str) -> str:
    """
    Minimal static player. Uses embedded 'const playlist = ...;'
    Expects each track has `full_path` like "tracks/<id>.mp3".
    """
    playlist_json = json.dumps(playlist_obj, ensure_ascii=False)
    # Keep it simple and robust: play/pause, next/prev, list clickable.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; line-height: 1.35; }}
    .row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    button {{ padding: 8px 12px; border: 1px solid #ccc; background: #fff; border-radius: 8px; cursor: pointer; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .box {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; max-width: 900px; }}
    .meta {{ color: #555; font-size: 13px; }}
    .list {{ margin-top: 12px; }}
    .item {{ padding: 8px 10px; border-radius: 10px; cursor: pointer; }}
    .item:hover {{ background: #f4f4f4; }}
    .item.active {{ background: #e8f0fe; }}
    audio {{ width: 100%; margin-top: 12px; }}
    code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="box">
    <h1 style="margin: 0 0 8px 0;">{title}</h1>
    <div class="meta" id="meta"></div>

    <div class="row" style="margin-top: 12px;">
      <button id="prevBtn">Prev</button>
      <button id="playBtn">Play</button>
      <button id="nextBtn">Next</button>
      <div class="meta" id="nowPlaying" style="margin-left: 8px;"></div>
    </div>

    <audio id="player" controls preload="metadata"></audio>

    <div class="list" id="list"></div>

    <div class="meta" style="margin-top: 12px;">
      Source: embedded playlist (<code>const playlist</code>) • Built: <span id="builtAt"></span>
    </div>
  </div>

  <script>
    const playlist = {playlist_json};
    const tracks = (playlist && playlist.tracks) ? playlist.tracks : [];
    const meta = document.getElementById("meta");
    const list = document.getElementById("list");
    const player = document.getElementById("player");
    const playBtn = document.getElementById("playBtn");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const nowPlaying = document.getElementById("nowPlaying");
    const builtAt = document.getElementById("builtAt");

    builtAt.textContent = new Date().toISOString();

    function trackTitle(t) {{
      return (t && (t.title || t.name || t.id)) || "Untitled";
    }}

    let idx = 0;

    function setButtons() {{
      prevBtn.disabled = (tracks.length === 0 || idx <= 0);
      nextBtn.disabled = (tracks.length === 0 || idx >= tracks.length - 1);
      playBtn.disabled = (tracks.length === 0);
    }}

    function setActive() {{
      const nodes = list.querySelectorAll(".item");
      nodes.forEach((n, i) => {{
        if (i === idx) n.classList.add("active");
        else n.classList.remove("active");
      }});
    }}

    function load(i, autoplay=false) {{
      if (!tracks.length) {{
        meta.textContent = "No tracks in playlist.";
        setButtons();
        return;
      }}
      idx = Math.max(0, Math.min(i, tracks.length - 1));
      const t = tracks[idx];
      const src = t.full_path || t.src || t.url || t.path || "";
      meta.textContent = `Playlist: ${{playlist.slug || playlist.id || "unknown"}} • Tracks: ${{tracks.length}}`;
      nowPlaying.textContent = `Now: ${{trackTitle(t)}} (${{idx+1}}/${{tracks.length}}) • ${{src}}`;
      player.src = src;
      setButtons();
      setActive();
      if (autoplay) {{
        player.play().catch(() => {{}});
      }}
    }}

    function renderList() {{
      list.innerHTML = "";
      tracks.forEach((t, i) => {{
        const div = document.createElement("div");
        div.className = "item";
        div.textContent = `${{i+1}}. ${{trackTitle(t)}}`;
        div.onclick = () => load(i, true);
        list.appendChild(div);
      }});
    }}

    playBtn.onclick = () => {{
      if (player.paused) player.play().catch(() => {{}});
      else player.pause();
    }};

    prevBtn.onclick = () => load(idx - 1, true);
    nextBtn.onclick = () => load(idx + 1, true);

    player.addEventListener("ended", () => {{
      if (idx < tracks.length - 1) load(idx + 1, true);
    }});

    renderList();
    load(0, false);
  </script>
</body>
</html>
"""


# ----------------------------
# Build command
# ----------------------------

def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(args.playlist)
    if not playlist_path.exists():
        raise SystemExit(f"[mgc.web] playlist not found: {playlist_path}")

    playlist_obj = _load_playlist(playlist_path)
    slug = _safe_slug(str(playlist_obj.get("slug") or playlist_obj.get("id") or args.slug or "tmp_wav_test"))

    out_root = Path(args.out_dir)
    site_dir = out_root / slug
    tracks_out = site_dir / "tracks"
    playlists_out = site_dir / "playlists"

    # Clean build dir (optional)
    if args.clean and site_dir.exists():
        shutil.rmtree(site_dir)

    tracks_out.mkdir(parents=True, exist_ok=True)
    playlists_out.mkdir(parents=True, exist_ok=True)

    # Copy playlist JSON into the site for debugging/inspection.
    shutil.copy2(playlist_path, playlists_out / f"{slug}.source.json")

    raw_tracks: List[TrackIn] = []
    for t in playlist_obj.get("tracks", []):
        if not isinstance(t, dict):
            continue
        ti = _coerce_track(t)
        if ti:
            raw_tracks.append(ti)

    kept_tracks: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Build each track: validate, copy/convert, then rewrite full_path to web-relative file.
    for ti in raw_tracks:
        src_fs = _resolve_track_source(playlist_path, ti.full_path)

        if not src_fs.exists():
            skipped.append({"id": ti.id, "reason": "missing", "src": str(src_fs)})
            LOG.warning("[web.build] skip missing track: id=%s src=%s", ti.id, src_fs)
            continue

        if not _ffprobe_ok(src_fs):
            skipped.append({"id": ti.id, "reason": "invalid_audio", "src": str(src_fs)})
            LOG.warning("[web.build] skip invalid audio (ffprobe failed): id=%s src=%s", ti.id, src_fs)
            continue

        # Decide output filename/ext.
        out_ext = ".mp3" if args.prefer_mp3 else src_fs.suffix.lower()
        if out_ext not in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"):
            # Normalize unknown audio containers to mp3 if prefer_mp3 else wav.
            out_ext = ".mp3" if args.prefer_mp3 else ".wav"

        out_name = f"{ti.id}{out_ext}"
        dst_fs = tracks_out / out_name

        ok = True
        if args.prefer_mp3:
            # Convert to mp3 always (even if src is mp3) to normalize.
            ok = _convert_to_mp3(src_fs, dst_fs)
            if not ok:
                skipped.append({"id": ti.id, "reason": "mp3_convert_failed", "src": str(src_fs)})
                LOG.warning("[web.build] skip mp3 convert failed: id=%s src=%s", ti.id, src_fs)
                continue
        else:
            # Copy as-is.
            dst_fs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fs, dst_fs)

        # Final sanity: make sure output is decodable too.
        if not _ffprobe_ok(dst_fs):
            skipped.append({"id": ti.id, "reason": "built_invalid_audio", "dst": str(dst_fs)})
            LOG.warning("[web.build] skip built audio invalid: id=%s dst=%s", ti.id, dst_fs)
            try:
                dst_fs.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue

        # Track object for the web player
        kept_tracks.append(
            {
                "id": ti.id,
                "title": ti.title,
                "artist": "MGC",
                "mood": "focus",
                "genre": "web",
                "bpm": 120,
                "duration_sec": 0,
                # The key the player uses:
                "full_path": f"tracks/{out_name}",
            }
        )

    # Rebuild embedded playlist deterministically-ish:
    # preserve slug; created_at from input if present else now.
    created_at = str(playlist_obj.get("created_at") or playlist_obj.get("created_ts") or _utc_now_iso())
    embedded_playlist = {
        "id": "tmp",
        "slug": slug,
        "created_at": created_at,
        "tracks": kept_tracks,
        "skipped": skipped if args.emit_skipped else [],
    }

    # Write helper JSONs (useful for debugging even though the player embeds).
    _write_json(tracks_out / "tracks.json", kept_tracks)
    _write_json(site_dir / "tracks.json", kept_tracks)
    _write_json(playlists_out / f"{slug}.json", embedded_playlist)

    # Write index.html
    title = args.title or f"MGC Web Player — {slug}"
    html = _render_index_html(playlist_obj=embedded_playlist, title=title)
    _write_text(site_dir / "index.html", html)

    if args.json:
        print(
            json.dumps(
                {
                    "out_dir": str(site_dir),
                    "slug": slug,
                    "tracks_kept": len(kept_tracks),
                    "tracks_skipped": len(skipped),
                    "prefer_mp3": bool(args.prefer_mp3),
                    "index_html": str(site_dir / "index.html"),
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(str(site_dir / "index.html"))
        LOG.info("[web.build] built: %s (kept=%d skipped=%d)", site_dir, len(kept_tracks), len(skipped))

    # CI-friendly: return non-zero only if everything got skipped (optional strictness)
    if args.fail_if_empty and not kept_tracks:
        return 2
    return 0


# ----------------------------
# Serve command (simple)
# ----------------------------

def cmd_web_serve(args: argparse.Namespace) -> int:
    """
    Convenience wrapper over python -m http.server.
    Serves from repo root by default so /data/web/... works.
    """
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[mgc.web] serve root not found: {root}")

    # Use stdlib http.server (no extra deps).
    # Note: we intentionally don't daemonize; caller controls lifecycle.
    import http.server
    import socketserver

    os.chdir(str(root))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", int(args.port)), handler) as httpd:
        LOG.info("[web.serve] serving %s on port %s", root, args.port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            LOG.info("[web.serve] stopped")
    return 0


# ----------------------------
# CLI registration
# ----------------------------

def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Build/serve the static web player")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    b = ws.add_parser("build", help="Build a static web player site")
    b.add_argument("--playlist", required=True, help="Path to playlist JSON")
    b.add_argument("--slug", default=None, help="Override slug (default from playlist)")
    b.add_argument("--title", default=None, help="HTML title")
    b.add_argument("--out-dir", default="data/web", help="Output directory root (default: data/web)")
    b.add_argument("--prefer-mp3", action="store_true", help="Convert/copy audio to MP3 for browser playback")
    b.add_argument("--clean", action="store_true", help="Delete output site dir before writing")
    b.add_argument("--emit-skipped", action="store_true", help="Include skipped list in embedded playlist JSON")
    b.add_argument("--fail-if-empty", action="store_true", help="Exit non-zero if no playable tracks were built")
    b.add_argument("--json", action="store_true", help="Output JSON summary")
    b.set_defaults(func=cmd_web_build)

    s = ws.add_parser("serve", help="Serve static files (defaults to repo root)")
    s.add_argument("--root", default=".", help="Directory to serve (default: .)")
    s.add_argument("--port", default=8000, type=int, help="Port (default: 8000)")
    s.set_defaults(func=cmd_web_serve)

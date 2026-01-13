from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


# Default output root for the static site
DEFAULT_WEB_ROOT = Path("data/web")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _safe_slug(s: str) -> str:
    s2 = (s or "").strip().replace(" ", "_")
    return s2 if s2 else "playlist"


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _transcode_to_mp3(src: Path, dst: Path) -> None:
    """
    Transcode any audio file to mp3 using ffmpeg.
    We keep it simple + browser-friendly:
      - 44.1kHz, stereo, 192k bitrate
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        str(dst),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")


def _copy_track_asset(
    *,
    src_full_path: Path,
    dst_dir: Path,
    track_id: str,
    prefer_mp3: bool,
) -> Tuple[str, str]:
    """
    Copies/transcodes a track into dst_dir.

    Returns:
      (rel_path_for_playlist, mime)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # If prefer_mp3 and ffmpeg exists, write mp3
    if prefer_mp3 and _ffmpeg_exists():
        dst = dst_dir / f"{track_id}.mp3"
        _transcode_to_mp3(src_full_path, dst)
        return (f"tracks/{dst.name}", "audio/mpeg")

    # Otherwise copy original extension as-is
    ext = src_full_path.suffix.lower() or ".bin"
    dst = dst_dir / f"{track_id}{ext}"
    shutil.copy2(src_full_path, dst)

    # Best-effort mime
    mime, _ = mimetypes.guess_type(str(dst))
    return (f"tracks/{dst.name}", mime or "application/octet-stream")


PLAYER_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MGC Web Player</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .card { max-width: 820px; border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #333; background: #fff; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .meta { color: #444; font-size: 14px; }
    .track { padding: 10px 0; border-top: 1px solid #eee; }
    .track:first-of-type { border-top: 0; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2 id="title">MGC Player</h2>
    <div class="meta" id="subtitle"></div>
    <div style="height: 12px"></div>

    <audio id="audio" controls preload="metadata" style="width: 100%"></audio>

    <div style="height: 12px"></div>
    <div class="row">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <div class="meta" id="now"></div>
    </div>

    <div style="height: 16px"></div>
    <div id="list"></div>
  </div>

<script>
async function main() {
  const playlistUrl = "./playlist.json";
  const res = await fetch(playlistUrl, { cache: "no-store" });
  if (!res.ok) {
    document.getElementById("subtitle").textContent = "Failed to load playlist.json: " + res.status;
    return;
  }
  const pl = await res.json();
  const tracks = (pl.tracks || []).map((t, idx) => ({...t, _idx: idx}));
  const slug = pl.slug || "playlist";
  document.getElementById("title").textContent = slug;
  document.getElementById("subtitle").textContent = "tracks: " + tracks.length;

  const audio = document.getElementById("audio");
  const now = document.getElementById("now");
  let i = 0;

  function setTrack(idx) {
    i = (idx + tracks.length) % tracks.length;
    const t = tracks[i];
    const src = t.web_path || t.full_path || t.preview_path;
    audio.src = src;
    now.textContent = "Now: " + (t.title || t.id || ("Track " + (i+1))) + " (" + src + ")";
  }

  function renderList() {
    const list = document.getElementById("list");
    list.innerHTML = "";
    tracks.forEach((t, idx) => {
      const div = document.createElement("div");
      div.className = "track";
      const name = (t.title || t.id || ("Track " + (idx+1)));
      const src = t.web_path || t.full_path || t.preview_path || "";
      div.innerHTML = "<div><strong>" + name + "</strong></div>" +
                      "<div class='meta'><code>" + src + "</code></div>";
      div.onclick = () => { setTrack(idx); audio.play().catch(()=>{}); };
      list.appendChild(div);
    });
  }

  document.getElementById("prev").onclick = () => { setTrack(i - 1); audio.play().catch(()=>{}); };
  document.getElementById("next").onclick = () => { setTrack(i + 1); audio.play().catch(()=>{}); };

  if (tracks.length === 0) {
    document.getElementById("subtitle").textContent = "No tracks in playlist.json";
    document.getElementById("prev").disabled = true;
    document.getElementById("next").disabled = true;
    return;
  }

  renderList();
  setTrack(0);

  audio.addEventListener("ended", () => {
    setTrack(i + 1);
    audio.play().catch(()=>{});
  });

  // Useful debugging
  audio.addEventListener("error", () => {
    console.log("audio error", audio.error);
  });
}
main();
</script>
</body>
</html>
"""


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(args.playlist)
    if not playlist_path.exists():
        raise SystemExit(f"[mgc.web] playlist not found: {playlist_path}")

    pl = _read_json(playlist_path)
    slug = _safe_slug(str(pl.get("slug") or playlist_path.stem))
    out_root = Path(args.out_dir) / slug
    tracks_dir = out_root / "tracks"

    out_root.mkdir(parents=True, exist_ok=True)

    # Copy/transcode tracks and rewrite playlist paths to relative web paths.
    rewritten = dict(pl)
    new_tracks: List[Dict[str, Any]] = []

    prefer_mp3 = bool(args.prefer_mp3)

    for t in (pl.get("tracks") or []):
        td = dict(t)
        tid = str(td.get("id") or td.get("track_id") or td.get("uuid") or "")
        if not tid:
            # fallback stable id
            tid = f"track_{len(new_tracks)+1}"

        # Choose source path: full_path first, else preview_path.
        src = td.get("full_path") or td.get("preview_path")
        if not src:
            td["web_path"] = ""
            new_tracks.append(td)
            continue

        src_path = Path(str(src))
        if not src_path.exists():
            # keep original but mark missing
            td["web_path"] = str(src)
            td["web_missing"] = True
            new_tracks.append(td)
            continue

        rel_path, mime = _copy_track_asset(
            src_full_path=src_path,
            dst_dir=tracks_dir,
            track_id=tid,
            prefer_mp3=prefer_mp3,
        )
        td["web_path"] = rel_path
        td["web_mime"] = mime

        # Optional: remove absolute filesystem paths from the served playlist for safety
        if args.strip_paths:
            td.pop("full_path", None)
            td.pop("preview_path", None)

        new_tracks.append(td)

    rewritten["slug"] = slug
    rewritten["tracks"] = new_tracks

    # Write build outputs
    _write_json(out_root / "playlist.json", rewritten)
    (out_root / "index.html").write_text(PLAYER_HTML, encoding="utf-8")

    # Simple build report
    report = {
        "slug": slug,
        "out_dir": str(out_root),
        "prefer_mp3": prefer_mp3,
        "ffmpeg_found": _ffmpeg_exists(),
        "track_count": len(new_tracks),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


class _StaticHandler(SimpleHTTPRequestHandler):
    """
    Serve static files with sane audio MIME types.
    """
    # Python's mimetypes can be inconsistent across platforms; force these.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".wave": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".json": "application/json",
    }


def cmd_web_serve(args: argparse.Namespace) -> int:
    root = Path(args.root_dir).resolve()
    if not root.exists():
        raise SystemExit(f"[mgc.web] root dir not found: {root}")

    # Change working dir so SimpleHTTPRequestHandler serves from root
    os.chdir(str(root))

    host = args.host
    port = int(args.port)

    httpd = ThreadingHTTPServer((host, port), _StaticHandler)
    print(f"[mgc.web] serving {root} on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def register_web_subcommand(subparsers) -> None:
    web = subparsers.add_parser("web", help="Build + serve a simple static web player")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    b = ws.add_parser("build", help="Build static web player for a playlist JSON")
    b.add_argument("--playlist", required=True, help="Path to a playlist JSON (with tracks[].full_path etc.)")
    b.add_argument("--out-dir", default=str(DEFAULT_WEB_ROOT), help="Output root directory (default: data/web)")
    b.add_argument("--prefer-mp3", action="store_true", help="Transcode to mp3 if ffmpeg is available")
    b.add_argument("--strip-paths", action="store_true", help="Remove absolute full_path/preview_path from served playlist.json")
    b.set_defaults(func=cmd_web_build)

    s = ws.add_parser("serve", help="Serve the built static player")
    s.add_argument("--root-dir", default=str(DEFAULT_WEB_ROOT), help="Directory to serve (default: data/web)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", default=8000, type=int)
    s.set_defaults(func=cmd_web_serve)

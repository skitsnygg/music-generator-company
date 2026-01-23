#!/usr/bin/env bash
set -euo pipefail

# Generates /var/lib/mgc/releases/feed.json by scanning:
#   /var/lib/mgc/releases/<release_id>/<context>/...
#   /var/lib/mgc/releases/<release_id>/web/<context>/...
# and latest as:
#   /var/lib/mgc/releases/latest -> /var/lib/mgc/releases/<release_id>
#
# Hardened:
# - flock to prevent concurrent writers
# - atomic feed.json write
# - tolerant of either /web/<context> or /<context> layout

ROOT_DIR="${ROOT_DIR:-/var/lib/mgc/releases}"
OUT_JSON="${OUT_JSON:-$ROOT_DIR/feed.json}"
LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/.feed.lock}"
MAX_ITEMS="${MAX_ITEMS:-200}"
BASE_URL="${BASE_URL:-}"   # optional, e.g. https://your-domain.example

log() { printf '%s %s\n' "[release_feed]" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v flock >/dev/null 2>&1 || die "flock not found"

[ -d "$ROOT_DIR" ] || die "ROOT_DIR not found: $ROOT_DIR"
mkdir -p "$(dirname "$OUT_JSON")"

# Lock
exec 9>"$LOCK_FILE"
flock -n 9 || die "another feed run is in progress (lock: $LOCK_FILE)"

tmp="$(mktemp "${OUT_JSON}.tmp.XXXXXX")"
cleanup() { rm -f "$tmp" || true; }
trap cleanup EXIT

log "ROOT_DIR=$ROOT_DIR"
log "OUT_JSON=$OUT_JSON"
log "MAX_ITEMS=$MAX_ITEMS"
log "BASE_URL=${BASE_URL:-<empty>}"

python3 - <<PY >"$tmp"
import json, os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(${ROOT_DIR@Q})
OUT = Path(${OUT_JSON@Q})
MAX_ITEMS = int(${MAX_ITEMS@Q})
BASE_URL = (${BASE_URL@Q}).strip().rstrip("/")

def iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def is_release_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    if name.startswith("."):
        return False
    if name in ("latest",):
        return False
    if name == OUT.name:
        return False
    return True

def looks_like_context_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    # minimal: one of these must exist
    return any((d / fn).exists() for fn in ("index.html", "playlist.json", "web_manifest.json"))

def collect_contexts(release_dir: Path):
    # Support both layouts:
    #   release/<context>
    #   release/web/<context>
    candidates = []
    web = release_dir / "web"
    if web.is_dir():
        for child in web.iterdir():
            if looks_like_context_dir(child):
                candidates.append(("web", child.name, child))
    for child in release_dir.iterdir():
        if child.name in ("web", "bundle", "marketing"):
            continue
        if looks_like_context_dir(child):
            candidates.append(("root", child.name, child))

    # de-dupe by context name, prefer /web/<context> if both exist
    by_ctx = {}
    for kind, ctx, path in candidates:
        prev = by_ctx.get(ctx)
        if prev is None or (prev[0] == "root" and kind == "web"):
            by_ctx[ctx] = (kind, path)

    out = []
    for ctx in sorted(by_ctx.keys()):
        kind, path = by_ctx[ctx]
        # choose URL prefix matching layout
        out.append({"context": ctx, "kind": kind})
    return out, by_ctx

def dir_mtime_iso(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    except Exception:
        return "1970-01-01T00:00:00Z"

def make_url(release_id: str, ctx: str, kind: str, latest: bool):
    if latest:
        # Nginx: user said it serves /latest/web and /releases.
        # If kind == web, use /latest/web/<ctx>/ else /latest/<ctx>/
        path = f"/latest/web/{ctx}/" if kind == "web" else f"/latest/{ctx}/"
    else:
        path = f"/releases/{release_id}/web/{ctx}/" if kind == "web" else f"/releases/{release_id}/{ctx}/"
    return f"{BASE_URL}{path}" if BASE_URL else path

def read_playlist_track_count(p: Path) -> int:
    pl = p / "playlist.json"
    if not pl.exists():
        return 0
    try:
        obj = json.loads(pl.read_text(encoding="utf-8"))
        tracks = obj.get("tracks") or []
        return len(tracks) if isinstance(tracks, list) else 0
    except Exception:
        return 0

def latest_entry(latest_path: Path):
    if not latest_path.exists():
        return {"ok": False, "reason": "latest_missing", "contexts": []}
    if not latest_path.is_dir():
        return {"ok": False, "reason": "latest_not_dir", "contexts": []}

    ctx_list, by_ctx = collect_contexts(latest_path)
    contexts = []
    for ctx in sorted(by_ctx.keys()):
        kind, path = by_ctx[ctx]
        contexts.append({
            "context": ctx,
            "kind": kind,
            "mtime": dir_mtime_iso(path),
            "track_count": read_playlist_track_count(path),
            "url": make_url("latest", ctx, kind, latest=True),
        })

    return {"ok": True, "contexts": contexts}

# Build releases list
release_dirs = [p for p in ROOT.iterdir() if is_release_dir(p)]
# deterministic scan order
release_dirs.sort(key=lambda p: p.name)

releases = []
for r in release_dirs:
    ctx_list, by_ctx = collect_contexts(r)
    if not by_ctx:
        continue

    contexts = []
    release_mtime = dir_mtime_iso(r)
    for ctx in sorted(by_ctx.keys()):
        kind, path = by_ctx[ctx]
        m = dir_mtime_iso(path)
        release_mtime = max(release_mtime, m)
        contexts.append({
            "context": ctx,
            "kind": kind,
            "mtime": m,
            "track_count": read_playlist_track_count(path),
            "url": make_url(r.name, ctx, kind, latest=False),
        })

    releases.append({
        "release_id": r.name,
        "mtime": release_mtime,
        "contexts": contexts,
        "url": (f"{BASE_URL}/releases/{r.name}/" if BASE_URL else f"/releases/{r.name}/"),
    })

# Sort newest-first by mtime then release_id for determinism
releases.sort(key=lambda e: (e["mtime"], e["release_id"]), reverse=True)
if MAX_ITEMS > 0:
    releases = releases[:MAX_ITEMS]

feed = {
    "schema_version": 1,
    "generated_at": iso_now(),
    "root_dir": str(ROOT),
    "latest": latest_entry(ROOT / "latest"),
    "releases": releases,
}

print(json.dumps(feed, indent=2, sort_keys=True) + "\n")
PY

# Validate JSON before replacing existing feed
python3 -m json.tool "$tmp" >/dev/null 2>&1 || die "generated feed.json is invalid JSON"

# Atomic replace
mv -f "$tmp" "$OUT_JSON"
chmod 0644 "$OUT_JSON" || true

log "OK wrote $OUT_JSON ($(wc -c <"$OUT_JSON" | tr -d ' ') bytes)"

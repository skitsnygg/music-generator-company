#!/usr/bin/env python3
import json
import os
import sys


def extract_path(t: dict) -> str:
    if not isinstance(t, dict):
        return ""
    for k in ("path", "file", "audio_path", "audio", "src"):
        v = t.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_playlist_tracks.py PLAYLIST_PATH OUT_DIR", file=sys.stderr)
        return 2

    playlist = sys.argv[1]
    out_dir = os.path.abspath(sys.argv[2])

    if not os.path.isfile(playlist):
        print(f"ERROR: playlist not found: {playlist}", file=sys.stderr)
        return 2

    with open(playlist, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tracks = obj.get("tracks") or []
    playlist_dir = os.path.dirname(playlist)
    missing = []

    for t in tracks:
        rel = extract_path(t)
        if not rel:
            continue

        resolved = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(playlist_dir, rel))
        resolved_abs = os.path.abspath(resolved)

        if not (resolved_abs == out_dir or resolved_abs.startswith(out_dir + os.sep)):
            missing.append(
                {
                    "track_id": t.get("id") if isinstance(t, dict) else None,
                    "path": rel,
                    "resolved": resolved_abs,
                    "reason": "path_outside_out_dir",
                }
            )
            continue

        if not os.path.exists(resolved_abs):
            missing.append(
                {
                    "track_id": t.get("id") if isinstance(t, dict) else None,
                    "path": rel,
                    "resolved": resolved_abs,
                    "reason": "missing_file",
                }
            )

    if missing:
        print("ERROR: invalid/missing track files:", file=sys.stderr)
        for m in missing[:200]:
            print(m, file=sys.stderr)
        return 2

    print("OK: playlist track files exist (and are inside OUT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

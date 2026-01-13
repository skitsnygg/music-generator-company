def cmd_web_build(args: argparse.Namespace) -> int:
    # HARD GUARANTEE: when --json is set, stdout must be JSON only.
    # If any logging handler is pointed at sys.stdout, json piping breaks.
    # So, in json mode, forcibly redirect *logging* stream handlers away from stdout.
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
                h.setStream(sys.stderr)

        # Also crank down log noise defensively.
        logging.getLogger("mgc.web").setLevel(logging.CRITICAL + 1)

    playlist_path = Path(args.playlist)
    if not playlist_path.exists():
        raise SystemExit(f"[mgc.web] playlist not found: {playlist_path}")

    playlist_obj = _load_playlist(playlist_path)
    slug = _safe_slug(str(playlist_obj.get("slug") or playlist_obj.get("id") or args.slug or "tmp_wav_test"))

    out_root = Path(args.out_dir)
    site_dir = out_root / slug
    tracks_out = site_dir / "tracks"
    playlists_out = site_dir / "playlists"

    if args.clean and site_dir.exists():
        shutil.rmtree(site_dir)

    tracks_out.mkdir(parents=True, exist_ok=True)
    playlists_out.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the source playlist for inspection/debugging.
    shutil.copy2(playlist_path, playlists_out / f"{slug}.source.json")

    raw_tracks: List[TrackIn] = []
    for t in playlist_obj.get("tracks", []):
        if isinstance(t, dict):
            ti = _coerce_track(t)
            if ti:
                raw_tracks.append(ti)

    kept_tracks: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

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

        out_ext = ".mp3" if args.prefer_mp3 else src_fs.suffix.lower()
        if out_ext not in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"):
            out_ext = ".mp3" if args.prefer_mp3 else ".wav"

        out_name = f"{ti.id}{out_ext}"
        dst_fs = tracks_out / out_name

        if args.prefer_mp3:
            if not _convert_to_mp3(src_fs, dst_fs):
                skipped.append({"id": ti.id, "reason": "mp3_convert_failed", "src": str(src_fs)})
                LOG.warning("[web.build] skip mp3 convert failed: id=%s src=%s", ti.id, src_fs)
                continue
        else:
            shutil.copy2(src_fs, dst_fs)

        if not _ffprobe_ok(dst_fs):
            skipped.append({"id": ti.id, "reason": "built_invalid_audio", "dst": str(dst_fs)})
            LOG.warning("[web.build] skip built audio invalid: id=%s dst=%s", ti.id, dst_fs)
            try:
                dst_fs.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue

        kept_tracks.append(
            {
                "id": ti.id,
                "title": ti.title,
                "full_path": f"tracks/{out_name}",
            }
        )

    embedded_playlist = {
        "id": "tmp",
        "slug": slug,
        "created_at": str(playlist_obj.get("created_at") or playlist_obj.get("created_ts") or _utc_now_iso()),
        "tracks": kept_tracks,
        "skipped": skipped if args.emit_skipped else [],
    }

    _write_json(tracks_out / "tracks.json", kept_tracks)
    _write_json(playlists_out / f"{slug}.json", embedded_playlist)

    title = args.title or f"MGC Web Player — {slug}"
    _write_text(site_dir / "index.html", _render_index_html(playlist_obj=embedded_playlist, title=title))

    summary = {
        "out_dir": str(site_dir),
        "slug": slug,
        "tracks_kept": len(kept_tracks),
        "tracks_skipped": len(skipped),
        "prefer_mp3": bool(args.prefer_mp3),
        "index_html": str(site_dir / "index.html"),
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        LOG.info("[web.build] built: %s (kept=%d skipped=%d)", site_dir, len(kept_tracks), len(skipped))
        print(str(site_dir / "index.html"))

    if args.fail_if_empty and not kept_tracks:
        return 2
    return 0

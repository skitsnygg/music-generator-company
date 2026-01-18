# src/mgc/marketing/publish.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _default_receipts_dir() -> Path:
    # Prefer env MGC_EVIDENCE_DIR if present; otherwise keep it simple and repo-friendly.
    import os

    base = Path(os.environ.get("MGC_EVIDENCE_DIR", "data/evidence")).expanduser()
    return base / "marketing" / "receipts"


@dataclass(frozen=True)
class PublishDeps:
    # time / determinism
    deterministic_now_iso: Callable[[bool], str]
    stable_uuid5: Callable[..., str]
    stable_json_dumps: Callable[[Any], str]

    # db
    db_connect: Callable[[str], Any]
    ensure_tables_minimal: Callable[[Any], None]
    db_marketing_posts_pending: Callable[[Any], Sequence[Any]]
    db_marketing_post_set_status: Callable[..., None]
    db_drop_mark_published: Callable[..., int]
    db_insert_event: Callable[..., None]

    # row helpers + content/meta resolution
    row_first: Callable[[Any, Sequence[str]], Any]
    marketing_row_meta: Callable[[Any, Any], Optional[Mapping[str, Any]]]
    marketing_row_content: Callable[[Any, Any], str]


def _safe_str(x: Any) -> str:
    return str(x or "")


def _write_receipt(
    *,
    deps: PublishDeps,
    receipts_dir: Path,
    receipt: Dict[str, Any],
) -> Path:
    """
    Append-only receipt writer.

    Path is deterministic by receipt_id, so re-runs won't generate duplicates.
    We do not overwrite existing receipts.
    """
    receipt_id = _safe_str(receipt.get("receipt_id"))
    platform = _safe_str(receipt.get("platform") or "unknown")

    # Bucket by platform; filename by receipt_id for idempotency.
    out_path = receipts_dir / platform / f"{receipt_id}.json"

    if out_path.exists():
        return out_path

    _atomic_write_text(out_path, deps.stable_json_dumps(receipt) + "\n")
    return out_path


def publish_marketing(
    *,
    deps: PublishDeps,
    db_path: str,
    limit: int = 50,
    dry_run: bool = False,
    deterministic: bool = False,
    write_receipts: bool = True,
    receipts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Publish pending marketing posts.

    Returns a JSON-serializable object for stable printing by the CLI.

    Side effects:
      - updates marketing post status (unless dry_run)
      - marks drops published (unless dry_run)
      - inserts marketing.published event (always)
      - writes receipts (unless write_receipts=False)
    """
    ts = deps.deterministic_now_iso(deterministic)

    con = deps.db_connect(db_path)
    deps.ensure_tables_minimal(con)

    pending = deps.db_marketing_posts_pending(con, limit=limit)

    def _first_id(r: Any) -> str:
        v = deps.row_first(r, ["id", "post_id", "marketing_post_id"])
        return _safe_str(v)

    def _first_created(r: Any) -> str:
        v = deps.row_first(r, ["created_at", "created_ts", "ts"])
        return _safe_str(v)

    pending_sorted = sorted(list(pending), key=lambda r: (_first_created(r), _first_id(r)))

    published: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    run_ids_touched: List[str] = []

    # Collect run_ids deterministically (based on the pending set, not publish side effects)
    for row in pending_sorted:
        meta = deps.marketing_row_meta(con, row) or {}
        rid = _safe_str(meta.get("run_id"))
        if rid:
            run_ids_touched.append(rid)
    run_ids_touched = sorted(set([r for r in run_ids_touched if r]))

    # Batch id: stable in deterministic mode; time-based in non-deterministic mode
    batch_id = deps.stable_uuid5(
        "marketing_publish_batch",
        (ts if not deterministic else "fixed"),
        str(limit),
        ("dry" if dry_run else "live"),
        ("|".join(run_ids_touched) if run_ids_touched else "no_runs"),
    )

    # Receipts
    _receipts_dir = (receipts_dir or _default_receipts_dir()).expanduser()
    receipt_paths: List[str] = []

    for row in pending_sorted:
        post_id = _first_id(row)
        platform = _safe_str(deps.row_first(row, ["platform", "channel", "destination"]) or "unknown")
        content = deps.marketing_row_content(con, row) or ""

        if not content.strip():
            skipped_ids.append(post_id)
            continue

        meta = deps.marketing_row_meta(con, row) or {}
        run_id = _safe_str(meta.get("run_id"))
        drop_id = _safe_str(meta.get("drop_id"))
        track_id = _safe_str(meta.get("track_id"))  # if present in meta

        publish_id = deps.stable_uuid5("publish", batch_id, post_id, platform)

        if not dry_run:
            deps.db_marketing_post_set_status(
                con,
                post_id=post_id,
                status="published",
                ts=ts,
                meta_patch={"published_id": publish_id, "published_ts": ts, "batch_id": batch_id},
            )

        item = {
            "post_id": post_id,
            "platform": platform,
            "published_id": publish_id,
            "published_ts": ts,
            "dry_run": dry_run,
            "content": content,
            "run_id": run_id,
            "drop_id": drop_id,
        }
        published.append(item)

        if write_receipts:
            status = "dry_run" if dry_run else "ok"
            receipt_id = deps.stable_uuid5(
                "marketing_receipt",
                batch_id,
                post_id,
                platform,
                status,
            )
            receipt = {
                "receipt_id": receipt_id,
                "batch_id": batch_id,
                "ts": ts,
                "status": status,
                "post_id": post_id,
                "platform": platform,
                "published_id": publish_id,
                "run_id": run_id,
                "drop_id": drop_id,
                "track_id": track_id,
                # Keep full copy for now; you can trim later if desired.
                "content": content,
            }
            p = _write_receipt(deps=deps, receipts_dir=_receipts_dir, receipt=receipt)
            receipt_paths.append(str(p))

    drops_updated: Dict[str, int] = {}
    if run_ids_touched and not dry_run:
        for rid in run_ids_touched:
            drops_updated[rid] = deps.db_drop_mark_published(
                con,
                run_id=rid,
                marketing_batch_id=batch_id,
                published_ts=ts,
            )

    deps.db_insert_event(
        con,
        event_id=deps.stable_uuid5("event", "marketing.published", batch_id),
        ts=ts,
        kind="marketing.published",
        actor="system",
        meta={
            "batch_id": batch_id,
            "count": len(published),
            "dry_run": dry_run,
            "skipped_empty": len(skipped_ids),
            "run_ids": run_ids_touched,
            "drops_updated": drops_updated,
            "receipts_written": len(receipt_paths) if write_receipts else 0,
            "receipts_dir": str(_receipts_dir) if write_receipts else "",
        },
    )

    out = {
        "batch_id": batch_id,
        "ts": ts,
        "count": len(published),
        "skipped_empty": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "run_ids": run_ids_touched,
        "drops_updated": drops_updated,
        "items": published,
    }
    if write_receipts:
        out["receipts_dir"] = str(_receipts_dir)
        out["receipts_written"] = len(receipt_paths)

    return out

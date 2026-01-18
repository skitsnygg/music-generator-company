# src/mgc/marketing/publish.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


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
    db_marketing_post_set_status: Callable[[Any], None]
    db_drop_mark_published: Callable[[Any], int]
    db_insert_event: Callable[[Any], None]

    # row helpers + content/meta resolution
    row_first: Callable[[Any, Sequence[str]], Any]
    marketing_row_meta: Callable[[Any, Any], Optional[Mapping[str, Any]]]
    marketing_row_content: Callable[[Any, Any], str]


def publish_marketing(
    *,
    deps: PublishDeps,
    db_path: str,
    limit: int = 50,
    dry_run: bool = False,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Publish pending marketing posts.

    Returns a JSON-serializable object for stable printing by the CLI.
    Side effects:
      - updates marketing post status (unless dry_run)
      - marks drops published (unless dry_run)
      - inserts marketing.published event (always)
    """
    ts = deps.deterministic_now_iso(deterministic)

    con = deps.db_connect(db_path)
    deps.ensure_tables_minimal(con)

    pending = deps.db_marketing_posts_pending(con, limit=limit)

    def _first_id(r: Any) -> str:
        v = deps.row_first(r, ["id", "post_id", "marketing_post_id"])
        return str(v or "")

    def _first_created(r: Any) -> str:
        v = deps.row_first(r, ["created_at", "created_ts", "ts"])
        return str(v or "")

    pending_sorted = sorted(list(pending), key=lambda r: (_first_created(r), _first_id(r)))

    published: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    run_ids_touched: List[str] = []

    # Collect run_ids deterministically (based on the pending set, not publish side effects)
    for row in pending_sorted:
        meta = deps.marketing_row_meta(con, row) or {}
        rid = str(meta.get("run_id") or "")
        if rid:
            run_ids_touched.append(rid)
    run_ids_touched = sorted(set([r for r in run_ids_touched if r]))

    batch_id = deps.stable_uuid5(
        "marketing_publish_batch",
        (ts if not deterministic else "fixed"),
        str(limit),
        ("dry" if dry_run else "live"),
        ("|".join(run_ids_touched) if run_ids_touched else "no_runs"),
    )

    for row in pending_sorted:
        post_id = _first_id(row)
        platform = str(deps.row_first(row, ["platform", "channel", "destination"]) or "unknown")
        content = deps.marketing_row_content(con, row)

        if not (content or "").strip():
            skipped_ids.append(post_id)
            continue

        meta = deps.marketing_row_meta(con, row) or {}
        run_id = str(meta.get("run_id") or "")
        drop_id = str(meta.get("drop_id") or "")

        publish_id = deps.stable_uuid5("publish", batch_id, post_id, platform)

        if not dry_run:
            deps.db_marketing_post_set_status(
                con,
                post_id=post_id,
                status="published",
                ts=ts,
                meta_patch={"published_id": publish_id, "published_ts": ts, "batch_id": batch_id},
            )

        published.append(
            {
                "post_id": post_id,
                "platform": platform,
                "published_id": publish_id,
                "published_ts": ts,
                "dry_run": dry_run,
                "content": content,
                "run_id": run_id,
                "drop_id": drop_id,
            }
        )

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
        },
    )

    return {
        "batch_id": batch_id,
        "ts": ts,
        "count": len(published),
        "skipped_empty": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "run_ids": run_ids_touched,
        "drops_updated": drops_updated,
        "items": published,
    }

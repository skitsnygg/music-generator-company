from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=None, separators=(",", ":"))


def _default_evidence_receipts_dir() -> Path:
    base = Path(os.environ.get("MGC_EVIDENCE_DIR", "data/evidence")).expanduser()
    return base / "marketing" / "receipts"


def _iter_receipt_files(receipts_root: Path) -> List[Path]:
    if not receipts_root.exists():
        return []
    # deterministic order
    return sorted([p for p in receipts_root.rglob("*.json") if p.is_file()])


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_marketing_index_from_receipts(
    *,
    receipts_root: Path,
    web_out_dir: Path,
    copy_receipts: bool = True,
) -> Dict[str, Any]:
    """
    Read evidence receipts and emit:
      - <web_out_dir>/marketing/marketing.json
      - optionally copy receipts under <web_out_dir>/marketing/receipts/**

    No DB access. Pure filesystem read/write.
    Deterministic output ordering.
    """
    receipts_root = receipts_root.expanduser()
    web_out_dir = web_out_dir.expanduser()
    marketing_dir = web_out_dir / "marketing"
    receipts_out_dir = marketing_dir / "receipts"
    marketing_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_receipt_files(receipts_root)

    items: List[Dict[str, Any]] = []
    batches: Dict[str, Dict[str, Any]] = {}

    for f in files:
        obj = _load_json(f)
        if not isinstance(obj, dict):
            continue

        batch_id = str(obj.get("batch_id") or "")
        ts = str(obj.get("ts") or "")
        platform = str(obj.get("platform") or "unknown")
        post_id = str(obj.get("post_id") or "")
        status = str(obj.get("status") or "")
        published_id = str(obj.get("published_id") or "")
        run_id = str(obj.get("run_id") or "")
        drop_id = str(obj.get("drop_id") or "")
        track_id = str(obj.get("track_id") or "")
        content = obj.get("content")

        # receipt path relative to receipts_root
        rel = f.relative_to(receipts_root).as_posix()
        web_receipt_rel = f"marketing/receipts/{rel}"

        item = {
            "batch_id": batch_id,
            "ts": ts,
            "platform": platform,
            "post_id": post_id,
            "status": status,
            "published_id": published_id,
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "content": content,
            "receipt_path": web_receipt_rel,
        }
        items.append(item)

        if batch_id:
            b = batches.get(batch_id)
            if b is None:
                b = {"batch_id": batch_id, "ts": ts, "items": []}
                batches[batch_id] = b
            # Keep batch ts stable: earliest ts wins
            if ts and (not b["ts"] or ts < b["ts"]):
                b["ts"] = ts
            b["items"].append(item)

        if copy_receipts:
            src = f
            dst = receipts_out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Deterministic ordering
    items_sorted = sorted(
        items,
        key=lambda x: (
            str(x.get("ts") or ""),
            str(x.get("batch_id") or ""),
            str(x.get("platform") or ""),
            str(x.get("post_id") or ""),
        ),
        reverse=True,  # newest first for UI
    )

    batches_list = list(batches.values())
    for b in batches_list:
        b["items"] = sorted(
            b["items"],
            key=lambda x: (
                str(x.get("platform") or ""),
                str(x.get("post_id") or ""),
            ),
        )
    batches_list = sorted(batches_list, key=lambda b: str(b.get("ts") or ""), reverse=True)

    out = {
        "schema": "mgc.web_marketing.v1",
        "receipts_root": str(receipts_root),
        "count": len(items_sorted),
        "batches": batches_list,
        "items": items_sorted,
    }

    (marketing_dir / "marketing.json").write_text(_stable_json_dumps(out) + "\n", encoding="utf-8")
    return out

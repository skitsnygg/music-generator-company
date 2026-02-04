from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class PublishError(RuntimeError):
    pass


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _format_media_base(base: str, mapping: Dict[str, str]) -> Tuple[str, bool]:
    out = base
    used = False
    for key, val in mapping.items():
        token = f"{{{key}}}"
        if token in out:
            out = out.replace(token, val)
            used = True
    return out, used


def build_post_text(payload: Dict[str, Any], platform: str) -> str:
    caption = str(payload.get("caption") or "").strip()
    if caption:
        return caption

    hook = str(payload.get("hook") or "").strip()
    title = str(payload.get("title") or "").strip()
    cta = str(payload.get("cta") or "").strip()
    parts = [p for p in (hook, title, cta) if p]
    text = " ".join(parts).strip()

    if not text:
        text = str(payload.get("text") or payload.get("message") or payload.get("content") or "").strip()

    if platform.lower() == "x":
        # Best-effort trim for X
        if len(text) > 280:
            text = text[:277].rstrip() + "..."

    return text


def pick_media_path(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("video_path", "media_path", "preview_path", "teaser_path", "asset_path", "cover_path"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _resolve_path(p: str, base_dir: Optional[Path]) -> Path:
    pp = Path(p)
    if not pp.is_absolute() and base_dir is not None:
        pp = (base_dir / pp).resolve()
    return pp


def _scheduler_url() -> Optional[str]:
    url = (os.environ.get("MGC_MARKETING_SCHEDULER_URL") or "").strip()
    return url or None


def _resolve_video_url(payload: Dict[str, Any], base_dir: Optional[Path]) -> Optional[str]:
    for key in ("video_url", "media_url", "public_url"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    base = (os.environ.get("MGC_MARKETING_MEDIA_BASE") or "").strip()
    if not base:
        return None

    media_path = pick_media_path(payload)
    if not media_path:
        return None

    p = _resolve_path(media_path, base_dir)
    if p.is_absolute():
        rel: str
        if base_dir is not None:
            try:
                rel = p.resolve().relative_to(base_dir.resolve()).as_posix()
            except Exception:
                rel = p.name
        else:
            rel = p.name
    else:
        rel = p.as_posix()

    media_file = Path(rel).name
    marketing_media_path = f"marketing/{rel.lstrip('/')}"

    period_key = str(payload.get("period_key") or "")
    if not period_key:
        period = payload.get("period")
        if isinstance(period, dict):
            period_key = str(period.get("label") or period.get("key") or "")

    mapping = {
        "context": str(payload.get("context") or ""),
        "schedule": str(payload.get("schedule") or ""),
        "period_key": period_key,
        "period": period_key,
        "drop_id": str(payload.get("drop_id") or ""),
        "track_id": str(payload.get("track_id") or ""),
        "media_path": rel,
        "media_rel": rel,
        "media_file": media_file,
        "media_name": media_file,
        "marketing_media_path": marketing_media_path,
    }

    media_tokens = {"media_path", "media_rel", "media_file", "media_name", "marketing_media_path"}
    has_media_token = any(f"{{{k}}}" in base for k in media_tokens)

    base = base.rstrip("/")
    formatted, _ = _format_media_base(base, mapping)
    if has_media_token:
        return formatted
    return f"{formatted}/{media_file}"


def publish(platform: str, *, payload: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    plat = (platform or "").strip().lower()
    if not plat:
        raise PublishError("platform missing")

    text = build_post_text(payload, plat)
    if not text:
        raise PublishError("post text is empty")

    media_path = pick_media_path(payload)

    if plat == "scheduler":
        sched = _scheduler_url()
        if not sched:
            raise PublishError("scheduler platform requested but MGC_MARKETING_SCHEDULER_URL not set")
        return _publish_scheduler(sched, platform=plat, payload=payload, text=text, base_dir=base_dir)

    if plat == "x":
        if _x_creds_present():
            return _publish_x(text=text, payload=payload, base_dir=base_dir)
        sched = _scheduler_url()
        if sched:
            return _publish_scheduler(sched, platform=plat, payload=payload, text=text, base_dir=base_dir)
        hook = (
            os.environ.get("MGC_MARKETING_WEBHOOK_X") or os.environ.get("MGC_MARKETING_WEBHOOK_URL")
        )
        if hook:
            return _publish_webhook(hook, platform=plat, payload=payload, text=text, media_path=media_path)
        raise PublishError(
            "missing X credentials (MGC_X_API_KEY, MGC_X_API_SECRET, MGC_X_ACCESS_TOKEN, MGC_X_ACCESS_TOKEN_SECRET) "
            "and no webhook configured for platform x"
        )

    if plat in ("youtube", "youtube_shorts", "yt"):
        return _publish_youtube(text=text, payload=payload, base_dir=base_dir)

    if plat in ("instagram", "instagram_reels", "ig", "reels"):
        return _publish_instagram_reels(text=text, payload=payload, base_dir=base_dir)

    if plat in ("tiktok", "tt"):
        return _publish_tiktok(text=text, payload=payload, base_dir=base_dir)

    # Scheduler fallback (credential-less)
    sched = _scheduler_url()
    if sched:
        return _publish_scheduler(sched, platform=plat, payload=payload, text=text, base_dir=base_dir)

    # Fallback: webhook per-platform or global
    hook = (
        os.environ.get(f"MGC_MARKETING_WEBHOOK_{plat.upper()}") or os.environ.get("MGC_MARKETING_WEBHOOK_URL")
    )
    if hook:
        return _publish_webhook(hook, platform=plat, payload=payload, text=text, media_path=media_path)

    raise PublishError(
        f"no publisher configured for platform={plat}. "
        "Set MGC_MARKETING_WEBHOOK_<PLATFORM> or MGC_MARKETING_WEBHOOK_URL, "
        "or configure X credentials for platform 'x'."
    )


# ---------------------------------------------------------------------------
# X (Twitter) publisher: OAuth 1.0a user context
# ---------------------------------------------------------------------------


def _oauth_percent_encode(s: str) -> str:
    return urllib.parse.quote(str(s), safe="~")


def _oauth1_header(
    url: str,
    method: str,
    *,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    nonce = uuid.uuid4().hex
    ts = str(int(time.time()))
    oauth_params = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": nonce,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": ts,
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    param_items = [(k, v) for k, v in oauth_params.items()]
    if extra_params:
        for k, v in extra_params.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                for item in v:
                    param_items.append((k, str(item)))
            else:
                param_items.append((k, str(v)))
    param_items.sort(key=lambda kv: (kv[0], kv[1]))
    param_str = "&".join(f"{_oauth_percent_encode(k)}={_oauth_percent_encode(v)}" for k, v in param_items)

    base = "&".join(
        [
            _oauth_percent_encode(method.upper()),
            _oauth_percent_encode(url),
            _oauth_percent_encode(param_str),
        ]
    )
    key = f"{_oauth_percent_encode(consumer_secret)}&{_oauth_percent_encode(token_secret)}"
    sig = base64.b64encode(hmac.new(key.encode("utf-8"), base.encode("utf-8"), hashlib.sha1).digest()).decode("utf-8")
    oauth_params["oauth_signature"] = sig

    header = "OAuth " + ", ".join(
        f'{_oauth_percent_encode(k)}="{_oauth_percent_encode(v)}"' for k, v in oauth_params.items()
    )
    return header


def _x_creds_present() -> bool:
    key = (os.environ.get("MGC_X_API_KEY") or "").strip()
    secret = (os.environ.get("MGC_X_API_SECRET") or "").strip()
    token = (os.environ.get("MGC_X_ACCESS_TOKEN") or "").strip()
    token_secret = (os.environ.get("MGC_X_ACCESS_TOKEN_SECRET") or "").strip()
    return bool(key and secret and token and token_secret)


def _publish_x(*, text: str, payload: Dict[str, Any], base_dir: Optional[Path]) -> Dict[str, Any]:
    key = (os.environ.get("MGC_X_API_KEY") or "").strip()
    secret = (os.environ.get("MGC_X_API_SECRET") or "").strip()
    token = (os.environ.get("MGC_X_ACCESS_TOKEN") or "").strip()
    token_secret = (os.environ.get("MGC_X_ACCESS_TOKEN_SECRET") or "").strip()

    media_id: Optional[str] = None
    media_path = pick_media_path(payload)
    if media_path:
        p = _resolve_path(media_path, base_dir)
        if not p.exists() or not p.is_file():
            raise PublishError(f"X media_path not found: {p}")
        media_id = _x_upload_media(
            p,
            consumer_key=key,
            consumer_secret=secret,
            token=token,
            token_secret=token_secret,
        )

    url = "https://api.x.com/2/tweets"
    payload_obj: Dict[str, Any] = {"text": text}
    if media_id:
        payload_obj["media"] = {"media_ids": [media_id]}
    body = json.dumps(payload_obj).encode("utf-8")
    headers = {
        "Authorization": _oauth1_header(
            url,
            "POST",
            consumer_key=key,
            consumer_secret=secret,
            token=token,
            token_secret=token_secret,
        ),
        "Content-Type": "application/json",
        "User-Agent": "MGC/1.0",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            try:
                obj = json.loads(data)
            except Exception:
                obj = {"raw": data}
            remote_id = None
            if isinstance(obj, dict):
                remote_id = (obj.get("data") or {}).get("id") if isinstance(obj.get("data"), dict) else obj.get("id")
            return {"platform": "x", "remote_id": remote_id, "raw": obj}
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"X publish failed: {e.code} {e.reason} {payload}") from e
    except Exception as e:
        raise PublishError(f"X publish failed: {e}") from e


# ---------------------------------------------------------------------------
# X media upload (v1.1)
# ---------------------------------------------------------------------------


def _x_guess_media(path: Path) -> Tuple[str, str, bool]:
    suf = path.suffix.lower()
    if suf in (".mp4",):
        return ("video/mp4", "tweet_video", True)
    if suf in (".mov",):
        return ("video/quicktime", "tweet_video", True)
    if suf in (".gif",):
        return ("image/gif", "tweet_gif", False)
    if suf in (".jpg", ".jpeg"):
        return ("image/jpeg", "tweet_image", False)
    if suf in (".png",):
        return ("image/png", "tweet_image", False)
    raise PublishError(f"unsupported X media type: {path.suffix}")


def _x_request(
    *,
    url: str,
    method: str,
    body: Optional[bytes],
    content_type: Optional[str],
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
    extra_params: Optional[Dict[str, Any]] = None,
    sign_url: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": _oauth1_header(
            sign_url or url,
            method,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            token=token,
            token_secret=token_secret,
            extra_params=extra_params,
        ),
        "User-Agent": "MGC/1.0",
    }
    if content_type:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"X media request failed: {e.code} {e.reason} {payload}") from e
    except Exception as e:
        raise PublishError(f"X media request failed: {e}") from e


def _multipart_form(
    *,
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> Tuple[bytes, str]:
    boundary = f"----mgc{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for key, val in fields.items():
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode("utf-8"))
        lines.append(b"")
        lines.append(str(val).encode("utf-8"))
    lines.append(f"--{boundary}".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode("utf-8")
    )
    lines.append(f"Content-Type: {content_type}".encode("utf-8"))
    lines.append(b"")
    lines.append(data)
    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _x_media_id_from(obj: Dict[str, Any]) -> str:
    mid = obj.get("media_id_string") if isinstance(obj, dict) else None
    if not mid:
        mid = str(obj.get("media_id") or "")
    return str(mid or "")


def _x_upload_media(
    path: Path,
    *,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> str:
    media_type, media_category, is_video = _x_guess_media(path)
    if is_video:
        return _x_upload_media_chunked(
            path,
            media_type=media_type,
            media_category=media_category,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            token=token,
            token_secret=token_secret,
        )

    data = path.read_bytes()
    body, ctype = _multipart_form(
        fields={"media_category": media_category},
        file_field="media",
        filename=path.name,
        content_type=media_type,
        data=data,
    )
    url = "https://upload.twitter.com/1.1/media/upload.json"
    obj = _x_request(
        url=url,
        method="POST",
        body=body,
        content_type=ctype,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        token=token,
        token_secret=token_secret,
        extra_params=None,
    )
    media_id = _x_media_id_from(obj)
    if not media_id:
        raise PublishError(f"X media upload failed: {obj}")
    return media_id


def _x_upload_media_chunked(
    path: Path,
    *,
    media_type: str,
    media_category: str,
    consumer_key: str,
    consumer_secret: str,
    token: str,
    token_secret: str,
) -> str:
    url = "https://upload.twitter.com/1.1/media/upload.json"
    total_bytes = path.stat().st_size

    init_params = {
        "command": "INIT",
        "total_bytes": str(total_bytes),
        "media_type": media_type,
        "media_category": media_category,
    }
    init_body = urllib.parse.urlencode(init_params).encode("utf-8")
    init_obj = _x_request(
        url=url,
        method="POST",
        body=init_body,
        content_type="application/x-www-form-urlencoded",
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        token=token,
        token_secret=token_secret,
        extra_params=init_params,
    )
    media_id = _x_media_id_from(init_obj)
    if not media_id:
        raise PublishError(f"X media INIT failed: {init_obj}")

    chunk_size = 4 * 1024 * 1024
    segment_index = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            body, ctype = _multipart_form(
                fields={
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": str(segment_index),
                },
                file_field="media",
                filename=path.name,
                content_type=media_type,
                data=chunk,
            )
            _x_request(
                url=url,
                method="POST",
                body=body,
                content_type=ctype,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                token=token,
                token_secret=token_secret,
                extra_params=None,
            )
            segment_index += 1

    finalize_params = {"command": "FINALIZE", "media_id": media_id}
    finalize_body = urllib.parse.urlencode(finalize_params).encode("utf-8")
    finalize_obj = _x_request(
        url=url,
        method="POST",
        body=finalize_body,
        content_type="application/x-www-form-urlencoded",
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        token=token,
        token_secret=token_secret,
        extra_params=finalize_params,
    )

    processing = finalize_obj.get("processing_info") if isinstance(finalize_obj, dict) else None
    if isinstance(processing, dict):
        state = processing.get("state")
        retries = 0
        while state in ("pending", "in_progress") and retries < 10:
            wait_s = int(processing.get("check_after_secs") or 2)
            time.sleep(max(wait_s, 1))
            status_params = {"command": "STATUS", "media_id": media_id}
            status_url = f"{url}?{urllib.parse.urlencode(status_params)}"
            status_obj = _x_request(
                url=status_url,
                method="GET",
                body=None,
                content_type=None,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                token=token,
                token_secret=token_secret,
                extra_params=status_params,
                sign_url=url,
            )
            processing = status_obj.get("processing_info") if isinstance(status_obj, dict) else None
            if not isinstance(processing, dict):
                break
            state = processing.get("state")
            if state == "failed":
                err = processing.get("error") if isinstance(processing.get("error"), dict) else None
                msg = err.get("message") if isinstance(err, dict) else "processing_failed"
                raise PublishError(f"X media processing failed: {msg}")
            retries += 1

    return media_id


# ---------------------------------------------------------------------------
# Webhook publisher (fallback)
# ---------------------------------------------------------------------------


def _publish_webhook(
    url: str,
    *,
    platform: str,
    payload: Dict[str, Any],
    text: str,
    media_path: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "platform": platform,
        "text": text,
        "payload": payload,
    }

    if media_path:
        body["media_path"] = media_path
        if _truthy_env("MGC_MARKETING_WEBHOOK_INCLUDE_MEDIA"):
            try:
                data = open(media_path, "rb").read()
                body["media_b64"] = base64.b64encode(data).decode("utf-8")
            except Exception:
                # best-effort only
                pass

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "MGC/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"raw": raw}
            remote_id = obj.get("id") if isinstance(obj, dict) else None
            return {"platform": platform, "remote_id": remote_id, "raw": obj}
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"Webhook publish failed: {e.code} {e.reason} {payload}") from e
    except Exception as e:
        raise PublishError(f"Webhook publish failed: {e}") from e


# ---------------------------------------------------------------------------
# Scheduler adapter (credential-less)
# ---------------------------------------------------------------------------


def _scheduler_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": "MGC/1.0"}
    auth = (os.environ.get("MGC_MARKETING_SCHEDULER_AUTH") or "").strip()
    if auth:
        if auth.lower().startswith(("bearer ", "token ", "basic ")):
            headers["Authorization"] = auth
        else:
            headers["Authorization"] = f"Bearer {auth}"

    extra = (os.environ.get("MGC_MARKETING_SCHEDULER_HEADERS") or "").strip()
    if extra:
        try:
            obj = json.loads(extra)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if v is None:
                        continue
                    headers[str(k)] = str(v)
        except Exception:
            pass
    return headers


def _publish_scheduler(
    url: str,
    *,
    platform: str,
    payload: Dict[str, Any],
    text: str,
    base_dir: Optional[Path],
) -> Dict[str, Any]:
    media_path = pick_media_path(payload)
    media_url = _resolve_video_url(payload, base_dir)

    post_id = str(payload.get("post_id") or payload.get("id") or "")
    idempotency = post_id
    if not idempotency:
        base = f"{platform}|{payload.get('drop_id') or ''}|{payload.get('track_id') or ''}|{text}"
        idempotency = uuid.uuid5(uuid.NAMESPACE_URL, base).hex

    period_key = str(payload.get("period_key") or "")
    if not period_key:
        period = payload.get("period")
        if isinstance(period, dict):
            period_key = str(period.get("label") or period.get("key") or "")

    body: Dict[str, Any] = {
        "adapter": "mgc.scheduler.v1",
        "platform": platform,
        "post_id": post_id or None,
        "idempotency_key": idempotency,
        "text": text,
        "title": payload.get("title"),
        "hook": payload.get("hook"),
        "cta": payload.get("cta"),
        "context": payload.get("context"),
        "schedule": payload.get("schedule"),
        "period_key": period_key or None,
        "drop_id": payload.get("drop_id"),
        "track_id": payload.get("track_id"),
        "media": {
            "url": media_url,
            "path": media_path,
        },
        "payload": payload,
    }

    if media_path and _truthy_env("MGC_MARKETING_SCHEDULER_INCLUDE_MEDIA"):
        try:
            data = _resolve_path(media_path, base_dir).read_bytes()
            body["media"]["b64"] = base64.b64encode(data).decode("utf-8")
        except Exception:
            pass

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_scheduler_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"raw": raw}
            remote_id = None
            if isinstance(obj, dict):
                remote_id = obj.get("id") or obj.get("schedule_id") or obj.get("post_id")
            return {"platform": platform, "remote_id": remote_id, "raw": obj}
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"Scheduler publish failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"Scheduler publish failed: {e}") from e


# ---------------------------------------------------------------------------
# YouTube Shorts (videos.insert via resumable upload)
# ---------------------------------------------------------------------------


def _publish_youtube(*, text: str, payload: Dict[str, Any], base_dir: Optional[Path]) -> Dict[str, Any]:
    token = (os.environ.get("MGC_YT_ACCESS_TOKEN") or "").strip()
    if not token:
        sched = _scheduler_url()
        if sched:
            return _publish_scheduler(sched, platform="youtube_shorts", payload=payload, text=text, base_dir=base_dir)
        raise PublishError("missing YouTube access token (MGC_YT_ACCESS_TOKEN)")

    media_path = pick_media_path(payload)
    if not media_path:
        raise PublishError("missing media_path/video_path for YouTube upload")

    p = _resolve_path(media_path, base_dir)
    if not p.exists() or not p.is_file():
        raise PublishError(f"YouTube media_path not found: {p}")

    title = str(payload.get("title") or text or "MGC Short")
    description = str(payload.get("description") or "")
    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    privacy = (os.environ.get("MGC_YT_PRIVACY") or "unlisted").strip()
    category = (os.environ.get("MGC_YT_CATEGORY_ID") or "").strip()

    meta = {
        "snippet": {
            "title": title,
            "description": description,
        },
        "status": {
            "privacyStatus": privacy,
        },
    }
    if tags:
        meta["snippet"]["tags"] = [str(t) for t in tags]
    if category:
        meta["snippet"]["categoryId"] = category

    meta_bytes = json.dumps(meta).encode("utf-8")
    init_url = "https://www.googleapis.com/upload/youtube/v3/videos?part=snippet,status&uploadType=resumable"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(p.stat().st_size),
    }

    init_resp, init_headers = _yt_request_json(
        init_url,
        data=meta_bytes,
        headers=headers,
        method="POST",
    )
    upload_url = init_headers.get("Location")

    if not upload_url:
        raise PublishError(f"YouTube init failed: missing upload Location header: {init_resp}")

    video_bytes = p.read_bytes()
    obj, _ = _yt_request_json(
        upload_url,
        data=video_bytes,
        headers={"Content-Type": "video/mp4", "Content-Length": str(len(video_bytes))},
        method="PUT",
    )

    video_id = None
    if isinstance(obj, dict):
        video_id = obj.get("id")
    if not video_id:
        raise PublishError(f"YouTube upload failed: missing id: {obj}")

    status_obj = _yt_poll_until_ready(video_id, token)
    return {"platform": "youtube_shorts", "remote_id": video_id, "raw": {"upload": obj, "status": status_obj}}


# ---------------------------------------------------------------------------
# YouTube helpers (retry + processing poll)
# ---------------------------------------------------------------------------


def _yt_retry_enabled() -> bool:
    raw = (os.environ.get("MGC_YT_RETRY") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _yt_retry_max() -> int:
    try:
        return int(os.environ.get("MGC_YT_RETRY_MAX") or "3")
    except Exception:
        return 3


def _yt_retry_sleep() -> float:
    try:
        return float(os.environ.get("MGC_YT_RETRY_SLEEP") or "2")
    except Exception:
        return 2.0


def _yt_should_retry(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


def _yt_request_json(
    url: str,
    *,
    data: Optional[bytes],
    headers: Dict[str, str],
    method: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    attempts = _yt_retry_max() if _yt_retry_enabled() else 1
    sleep_s = _yt_retry_sleep()
    last_exc: Optional[Exception] = None

    for i in range(attempts):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = {"raw": raw}
                return obj, dict(resp.headers)
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            last_exc = e
            code = int(getattr(e, "code", 0) or 0)
            payload_txt = e.read().decode("utf-8", errors="replace")
            if _yt_should_retry(code) and i + 1 < attempts:
                time.sleep(max(sleep_s * (2 ** i), 0.5))
                continue
            raise PublishError(f"YouTube request failed: {code} {e.reason} {payload_txt}") from e
        except Exception as e:
            last_exc = e
            if i + 1 < attempts and _yt_retry_enabled():
                time.sleep(max(sleep_s * (2 ** i), 0.5))
                continue
            raise PublishError(f"YouTube request failed: {e}") from e

    if last_exc:
        raise PublishError(f"YouTube request failed: {last_exc}") from last_exc
    raise PublishError("YouTube request failed")


def _yt_poll_until_ready(video_id: str, token: str) -> Optional[Dict[str, Any]]:
    raw = (os.environ.get("MGC_YT_POLL") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return None

    max_attempts = int(os.environ.get("MGC_YT_POLL_MAX") or "12")
    sleep_s = float(os.environ.get("MGC_YT_POLL_SLEEP") or "5")

    last: Optional[Dict[str, Any]] = None
    for _ in range(max_attempts):
        last = _yt_fetch_status(video_id, token)
        status, reason = _yt_processing_status(last)
        if status in ("processed", "succeeded"):
            return last
        if status in ("failed", "rejected", "terminated"):
            raise PublishError(f"YouTube processing failed: {reason or status}")
        time.sleep(max(sleep_s, 1.0))

    raise PublishError("YouTube processing not finished; retry later or disable polling with MGC_YT_POLL=0")


def _yt_fetch_status(video_id: str, token: str) -> Dict[str, Any]:
    qs = urllib.parse.urlencode({"part": "status,processingDetails", "id": video_id})
    url = f"https://www.googleapis.com/youtube/v3/videos?{qs}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    obj, _ = _yt_request_json(url, data=None, headers=headers, method="GET")
    return obj


def _yt_processing_status(obj: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not isinstance(obj, dict):
        return ("", "")
    items = obj.get("items") if isinstance(obj.get("items"), list) else []
    if not items:
        return ("", "missing_items")

    item = items[0] if isinstance(items[0], dict) else {}
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    processing = item.get("processingDetails") if isinstance(item.get("processingDetails"), dict) else {}

    upload_status = str(status.get("uploadStatus") or "").lower()
    proc_status = str(processing.get("processingStatus") or "").lower()
    reason = str(status.get("rejectionReason") or status.get("failureReason") or "")

    if proc_status:
        return (proc_status, reason)
    return (upload_status, reason)

# ---------------------------------------------------------------------------
# Instagram Reels (Graph API)
# ---------------------------------------------------------------------------


def _publish_instagram_reels(*, text: str, payload: Dict[str, Any], base_dir: Optional[Path]) -> Dict[str, Any]:
    token = (os.environ.get("MGC_IG_ACCESS_TOKEN") or "").strip()
    ig_user_id = (os.environ.get("MGC_IG_USER_ID") or "").strip()
    if not (token and ig_user_id):
        sched = _scheduler_url()
        if sched:
            return _publish_scheduler(sched, platform="instagram_reels", payload=payload, text=text, base_dir=base_dir)
        raise PublishError("missing Instagram credentials (MGC_IG_ACCESS_TOKEN, MGC_IG_USER_ID)")

    video_url = _resolve_video_url(payload, base_dir)
    if not video_url:
        raise PublishError("missing video_url/media_url for Instagram Reels (requires publicly reachable URL)")

    caption = text
    api_base = os.environ.get("MGC_IG_API_BASE", "https://graph.facebook.com/v19.0").rstrip("/")

    # 1) Create media container
    create_url = f"{api_base}/{ig_user_id}/media"
    create_body = urllib.parse.urlencode(
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(os.environ.get("MGC_IG_SHARE_TO_FEED", "false")).lower(),
            "access_token": token,
        }
    ).encode("utf-8")
    req = urllib.request.Request(create_url, data=create_body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"Instagram create failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"Instagram create failed: {e}") from e

    creation_id = obj.get("id") if isinstance(obj, dict) else None
    if not creation_id:
        raise PublishError(f"Instagram create failed: no creation id: {obj}")

    _ig_poll_until_ready(api_base, creation_id, token)

    # 2) Publish
    publish_url = f"{api_base}/{ig_user_id}/media_publish"
    publish_body = urllib.parse.urlencode(
        {"creation_id": creation_id, "access_token": token}
    ).encode("utf-8")
    req2 = urllib.request.Request(publish_url, data=publish_body, method="POST")
    try:
        with urllib.request.urlopen(req2, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            pub = json.loads(raw)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"Instagram publish failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"Instagram publish failed: {e}") from e

    remote_id = pub.get("id") if isinstance(pub, dict) else None
    return {"platform": "instagram_reels", "remote_id": remote_id, "raw": pub}


def _ig_poll_until_ready(api_base: str, creation_id: str, token: str) -> None:
    raw = (os.environ.get("MGC_IG_POLL") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return

    max_attempts = int(os.environ.get("MGC_IG_POLL_MAX") or "10")
    sleep_s = float(os.environ.get("MGC_IG_POLL_SLEEP") or "3")

    for _ in range(max_attempts):
        status = _ig_fetch_status(api_base, creation_id, token)
        if status == "FINISHED":
            return
        if status in ("ERROR", "FAILED"):
            raise PublishError(f"Instagram processing failed: status={status}")
        time.sleep(max(sleep_s, 0.5))

    raise PublishError("Instagram processing not finished; try again later or disable polling with MGC_IG_POLL=0")


def _ig_fetch_status(api_base: str, creation_id: str, token: str) -> str:
    params = urllib.parse.urlencode({"fields": "status_code", "access_token": token})
    url = f"{api_base}/{creation_id}?{params}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"Instagram status failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"Instagram status failed: {e}") from e

    if isinstance(obj, dict):
        return str(obj.get("status_code") or obj.get("status") or "").upper()
    return ""


# ---------------------------------------------------------------------------
# TikTok Content Posting API (pull-from-URL)
# ---------------------------------------------------------------------------


def _publish_tiktok(*, text: str, payload: Dict[str, Any], base_dir: Optional[Path]) -> Dict[str, Any]:
    token = (os.environ.get("MGC_TIKTOK_ACCESS_TOKEN") or "").strip()
    if not token:
        sched = _scheduler_url()
        if sched:
            return _publish_scheduler(sched, platform="tiktok", payload=payload, text=text, base_dir=base_dir)
        raise PublishError("missing TikTok access token (MGC_TIKTOK_ACCESS_TOKEN)")

    video_url = _resolve_video_url(payload, base_dir)
    if not video_url:
        raise PublishError("missing video_url/media_url for TikTok (requires publicly reachable URL)")

    privacy = (os.environ.get("MGC_TIKTOK_PRIVACY_LEVEL") or "PUBLIC").strip()
    api_base = os.environ.get("MGC_TIKTOK_API_BASE", "https://open.tiktokapis.com").rstrip("/")

    body = {
        "post_info": {
            "title": text,
            "privacy_level": privacy,
        },
        "source_info": {
            "source": "PULL_FROM_URL",
            "video_url": video_url,
        },
        "post_mode": "DIRECT_POST",
    }

    data = json.dumps(body).encode("utf-8")
    url = f"{api_base}/v2/post/publish/video/init/"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"TikTok init failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"TikTok init failed: {e}") from e

    data_obj = obj.get("data") if isinstance(obj, dict) else None
    publish_id = data_obj.get("publish_id") if isinstance(data_obj, dict) else None
    if not publish_id:
        raise PublishError(f"TikTok init failed: missing publish_id: {obj}")

    status_obj = _tiktok_poll_until_ready(api_base, token, str(publish_id))
    remote_id = str(publish_id)
    if isinstance(status_obj, dict):
        data_status = status_obj.get("data") if isinstance(status_obj.get("data"), dict) else None
        if isinstance(data_status, dict):
            post_ids = data_status.get("publicaly_available_post_id")
            if isinstance(post_ids, list) and post_ids:
                remote_id = str(post_ids[0])
    return {"platform": "tiktok", "remote_id": remote_id, "raw": {"init": obj, "status": status_obj}}


def _tiktok_poll_until_ready(api_base: str, token: str, publish_id: str) -> Optional[Dict[str, Any]]:
    raw = (os.environ.get("MGC_TIKTOK_POLL") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return None

    max_attempts = int(os.environ.get("MGC_TIKTOK_POLL_MAX") or "10")
    sleep_s = float(os.environ.get("MGC_TIKTOK_POLL_SLEEP") or "3")

    last: Optional[Dict[str, Any]] = None
    for _ in range(max_attempts):
        last = _tiktok_fetch_status(api_base, token, publish_id)
        status = _tiktok_status(last)
        if status in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"):
            return last
        if status == "FAILED":
            fail_reason = _tiktok_fail_reason(last)
            raise PublishError(f"TikTok publish failed: {fail_reason or 'failed'}")
        time.sleep(max(sleep_s, 0.5))

    raise PublishError("TikTok processing not finished; retry later or disable polling with MGC_TIKTOK_POLL=0")


def _tiktok_fetch_status(api_base: str, token: str, publish_id: str) -> Dict[str, Any]:
    url = f"{api_base}/v2/post/publish/status/fetch/"
    body = json.dumps({"publish_id": publish_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        payload_txt = e.read().decode("utf-8", errors="replace")
        raise PublishError(f"TikTok status failed: {e.code} {e.reason} {payload_txt}") from e
    except Exception as e:
        raise PublishError(f"TikTok status failed: {e}") from e

    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        code = str(err.get("code") or "").lower()
        if code and code != "ok":
            msg = err.get("message") or ""
            raise PublishError(f"TikTok status error: {code} {msg}".strip())
    return obj


def _tiktok_status(obj: Optional[Dict[str, Any]]) -> str:
    if not isinstance(obj, dict):
        return ""
    data = obj.get("data")
    if isinstance(data, dict):
        return str(data.get("status") or "").upper()
    return ""


def _tiktok_fail_reason(obj: Optional[Dict[str, Any]]) -> str:
    if not isinstance(obj, dict):
        return ""
    data = obj.get("data")
    if isinstance(data, dict):
        return str(data.get("fail_reason") or "")
    return ""

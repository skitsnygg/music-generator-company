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
from typing import Any, Dict, Optional


class PublishError(RuntimeError):
    pass


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


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
    for key in ("media_path", "preview_path", "teaser_path", "asset_path", "cover_path"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def publish(platform: str, *, payload: Dict[str, Any]) -> Dict[str, Any]:
    plat = (platform or "").strip().lower()
    if not plat:
        raise PublishError("platform missing")

    text = build_post_text(payload, plat)
    if not text:
        raise PublishError("post text is empty")

    media_path = pick_media_path(payload)

    if plat == "x":
        if _x_creds_present():
            return _publish_x(text=text)
        hook = (
            os.environ.get("MGC_MARKETING_WEBHOOK_X") or os.environ.get("MGC_MARKETING_WEBHOOK_URL")
        )
        if hook:
            return _publish_webhook(hook, platform=plat, payload=payload, text=text, media_path=media_path)
        raise PublishError(
            "missing X credentials (MGC_X_API_KEY, MGC_X_API_SECRET, MGC_X_ACCESS_TOKEN, MGC_X_ACCESS_TOKEN_SECRET) "
            "and no webhook configured for platform x"
        )

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


def _oauth1_header(url: str, method: str, *, consumer_key: str, consumer_secret: str, token: str, token_secret: str) -> str:
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

    # No extra params for JSON body
    param_items = [(k, v) for k, v in oauth_params.items()]
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


def _publish_x(*, text: str) -> Dict[str, Any]:
    key = (os.environ.get("MGC_X_API_KEY") or "").strip()
    secret = (os.environ.get("MGC_X_API_SECRET") or "").strip()
    token = (os.environ.get("MGC_X_ACCESS_TOKEN") or "").strip()
    token_secret = (os.environ.get("MGC_X_ACCESS_TOKEN_SECRET") or "").strip()

    url = "https://api.twitter.com/2/tweets"
    body = json.dumps({"text": text}).encode("utf-8")
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

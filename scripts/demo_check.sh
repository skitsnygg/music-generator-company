#!/usr/bin/env bash
set -euo pipefail

echo "[demo_check] starting full demo verification"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEED_PATH="/var/lib/mgc/releases/feed.json"
FEED_URL="http://127.0.0.1/releases/feed.json"

cd "$REPO_ROOT"

echo "[demo_check] repo: $REPO_ROOT"

# 1) Run daily pipeline (this regenerates latest + feed)
echo "[demo_check] running daily pipeline..."
sudo -E scripts/run_daily.sh

# 2) Verify feed exists on disk
echo "[demo_check] checking feed on disk..."
test -s "$FEED_PATH"
ls -la "$FEED_PATH"

# 3) Verify feed JSON is valid
echo "[demo_check] validating feed JSON..."
python3 -m json.tool "$FEED_PATH" >/dev/null
echo "[demo_check] feed json ok"

# 4) Verify nginx serves the feed
echo "[demo_check] fetching feed via nginx..."
curl -fsS "$FEED_URL" | python3 -m json.tool >/dev/null
echo "[demo_check] nginx serving feed ok"

# 5) Verify contexts are filtered (no .bak, no run)
echo "[demo_check] verifying context filtering..."
python3 - <<'PY'
import json
p="/var/lib/mgc/releases/feed.json"
o=json.load(open(p,"r",encoding="utf-8"))
names=[c["context"] for c in o["latest"]["contexts"]]
print("[demo_check] latest contexts:", names)
assert all(".bak." not in n for n in names), "backup contexts present"
assert "run" not in names, "run context present"
print("[demo_check] context filtering ok")
PY

# 6) Determinism proof: regenerate feed and compare content hash
echo "[demo_check] verifying content determinism..."
python3 - <<'PY'
import json, hashlib, subprocess, time
from pathlib import Path

p=Path("/var/lib/mgc/releases/feed.json")

def content_hash(obj):
    o=dict(obj)
    o.pop("generated_at", None)
    o.pop("content_sha256", None)
    canon=json.dumps(o, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()

a=json.loads(p.read_text(encoding="utf-8"))
h1=content_hash(a)

# regenerate feed (per-context, matches prod usage)
for c in ("focus","sleep","workout"):
    subprocess.check_call(["sudo","-E","scripts/publish_release_feed.sh","--context",c])

b=json.loads(p.read_text(encoding="utf-8"))
h2=content_hash(b)

print("[demo_check] content_sha256_1:", h1)
print("[demo_check] content_sha256_2:", h2)
assert h1 == h2, "content hash changed"
print("[demo_check] determinism ok")
PY

echo "[demo_check] ALL CHECKS PASSED"

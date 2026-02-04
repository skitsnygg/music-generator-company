#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from subprocess import run


class _Handler(BaseHTTPRequestHandler):
    max_req = 4
    count = 0
    out_path = Path("/tmp/mgc_scheduler_body.jsonl")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("ab") as f:
            f.write(body + b"\n")

        _Handler.count += 1
        payload = json.dumps({"schedule_id": f"scheduler_test_{_Handler.count}"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

        if _Handler.count >= self.max_req:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format, *args):
        return


def _find_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _load_drop_id(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return str(obj.get("drop_id") or "")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    evidence = repo / "data" / "evidence" / "focus" / "drop_evidence.json"
    if not evidence.exists():
        print(f"[marketing_scheduler_smoke] missing: {evidence}")
        return 2

    drop_id = _load_drop_id(evidence)
    if not drop_id:
        print("[marketing_scheduler_smoke] drop_id missing from evidence")
        return 2

    # Start a local scheduler server
    port = _find_port()
    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    _Handler.out_path.unlink(missing_ok=True)  # type: ignore[call-arg]

    env = os.environ.copy()
    env["MGC_MARKETING_SCHEDULER_URL"] = f"http://127.0.0.1:{port}/"

    cmd = [
        sys.executable,
        "-m",
        "mgc.main",
        "run",
        "publish-marketing",
        "--bundle-dir",
        str(repo / "data" / "evidence" / "focus" / "drop_bundle"),
        "--out-dir",
        str(repo / "data" / "evidence" / "focus"),
        "--publish-live",
        "--drop-id",
        drop_id,
    ]
    p = run(cmd, env=env)
    server.shutdown()
    t.join(timeout=5)

    if p.returncode != 0:
        print(f"[marketing_scheduler_smoke] publish-marketing failed rc={p.returncode}")
        return p.returncode

    if not _Handler.out_path.exists():
        print("[marketing_scheduler_smoke] scheduler body not captured")
        return 2

    print(f"[marketing_scheduler_smoke] ok saved={_Handler.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

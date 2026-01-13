from __future__ import annotations

import json
from typing import Any, Dict

from .base import GenerateRequest, GenerateResult


class StubProvider:
    name = "stub"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        payload: Dict[str, Any] = {
            "provider": self.name,
            "track_id": req.track_id,
            "run_id": req.run_id,
            "context": req.context,
            "seed": req.seed,
            "ts": req.ts,
            "deterministic": req.deterministic,
            "prompt": req.prompt,
        }
        b = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return GenerateResult(
            provider=self.name,
            artifact_bytes=b,
            mime="application/json",
            ext=".json",
            meta={"provider": self.name, "context": req.context, "seed": req.seed},
        )

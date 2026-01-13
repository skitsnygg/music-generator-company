from __future__ import annotations

from mgc.providers.base import GenerateRequest, GenerateResult


class DiffSingerProvider:
    name = "diffsinger"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        raise RuntimeError("DiffSinger provider is staged but not configured. Wire a local runner later.")

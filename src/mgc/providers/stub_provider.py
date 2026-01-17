from __future__ import annotations

"""Back-compat module name.

Some code paths may import mgc.providers.stub_provider.StubProvider.
The real implementation lives in mgc.providers.stub.
"""

from .stub import StubProvider

__all__ = ["StubProvider"]

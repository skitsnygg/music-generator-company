from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Entitlement:
    plan: str  # "free", "supporter", "pro"
    active: bool = True


class BillingAgent:
    """
    MVP billing agent.

    - Keeps an in-memory entitlement map (CI-safe, deterministic).
    - Later replace with Stripe + DB/webhook-driven entitlements.
    """

    def __init__(self) -> None:
        self._entitlements: Dict[str, Entitlement] = {}

    def set_entitlement(self, user_id: str, plan: str, active: bool = True) -> None:
        self._entitlements[str(user_id)] = Entitlement(plan=str(plan), active=bool(active))

    def is_paid_user(self, user_id: str) -> bool:
        ent = self._entitlements.get(str(user_id))
        if not ent:
            return False
        if not ent.active:
            return False
        return ent.plan in ("supporter", "pro")

from __future__ import annotations

class BillingAgent:
    """
    MVP stub: access control is not enforced yet.
    Later: Stripe subscription + webhook-driven entitlement checks.
    """
    def __init__(self) -> None:
        pass

    def is_paid_user(self, user_id: str) -> bool:
        return False

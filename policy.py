"""
Policy engine — deterministic, pure, zero LLM calls.

ARCHITECTURAL INVARIANT
-----------------------
No function in this module may call a model, read a clock, or touch the network.
Same input -> same output, forever. This is what makes the money questions
(is she owed a refund? how many hours of hotel?) *provably* correct under
`pytest` instead of "looked right in the chat transcript".

The LLM's only job downstream is to *phrase* the PolicyResult produced here.
It never computes, adjusts, or re-derives any value in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data import (
    CANCELLATION_REBOOKING_WINDOW_HOURS,
    DELAY_COMPENSATION_TIERS,
    FARE_DIFFERENCE_SUPERVISOR_THRESHOLD_RUPEES,
    MEAL_VOUCHER_VALUE_RUPEES,
    PRIORITY_REBOOKING_TIERS,
    REFUND_PROCESSING_BUSINESS_DAYS,
)


@dataclass(frozen=True)
class PolicyResult:
    """
    The single policy object every downstream layer reads.

    Frozen on purpose: once the deterministic layer has decided, no LLM node,
    UI callback, or graph edge can mutate an entitlement on its way to the
    customer. Downstream code reads; it does not write.
    """

    disruption_type: str  # "cancelled" | "delayed" | "unaffected" | "unknown"
    entitlements: tuple[str, ...] = ()
    customer_facts: dict[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    priority_rebooking: bool = False
    assumptions: tuple[str, ...] = ()

    def as_facts(self) -> dict[str, Any]:
        """Flat, JSON-safe dict handed to the reply-composing LLM node.

        This is the model's ENTIRE permitted knowledge of the customer's
        entitlements. Anything absent here must be answered with "I'll need to
        have that looked into" — see llm_nodes.COMPOSE_SYSTEM_PROMPT.
        """
        return {
            "disruption_type": self.disruption_type,
            "entitlements": list(self.entitlements),
            "allowed_actions": list(self.allowed_actions),
            "priority_rebooking": self.priority_rebooking,
            "notes": list(self.notes),
            **self.customer_facts,
        }


# --------------------------------------------------------------------------- #
# Rule family 1 — Cancellation Rebooking Rule
# --------------------------------------------------------------------------- #
def resolve_cancellation(booking: dict[str, Any]) -> PolicyResult:
    """
    "If a flight is cancelled by the airline, the customer is entitled to a free
    rebooking on the next available flight within 24 hours, OR a full refund,
    customer's choice."

    Note the entitlement is a CHOICE, not both. The agent must offer both and
    let the customer pick — it must never pre-decide for them.
    """
    if not booking.get("airline_caused", False):
        return PolicyResult(
            disruption_type="cancelled",
            customer_facts={"flight": booking.get("flight"), "route": booking.get("route")},
            notes=("Cancellation was not airline-caused; standard entitlements do not apply.",),
        )

    return PolicyResult(
        disruption_type="cancelled",
        entitlements=("free_rebooking_within_24h", "full_refund"),
        customer_facts={
            "flight": booking["flight"],
            "route": booking["route"],
            "scheduled_departure": booking["scheduled_departure"],
            "cancellation_reason": booking["status_detail"],
            "rebooking_window_hours": CANCELLATION_REBOOKING_WINDOW_HOURS,
            "refund_processing_business_days": REFUND_PROCESSING_BUSINESS_DAYS,
            "refund_destination": "original payment method only",
        },
        allowed_actions=(
            "Rebook on the next available flight within 24 hours at no charge",
            "Initiate a full refund to the original payment method",
        ),
        notes=(
            "Customer chooses ONE: free rebooking within 24 hours, or a full refund.",
            f"Refunds are processed in full within {REFUND_PROCESSING_BUSINESS_DAYS} "
            "business days, to the original payment method only.",
        ),
    )


# --------------------------------------------------------------------------- #
# Rule family 2 — Delay Compensation Rule
# --------------------------------------------------------------------------- #
def _select_delay_tier(delay_hours: float) -> dict[str, Any]:
    """Pick the tier whose (min_exclusive, max_inclusive] band contains the delay."""
    for tier in DELAY_COMPENSATION_TIERS:
        lower_ok = tier["min_exclusive"] is None or delay_hours > tier["min_exclusive"]
        upper_ok = tier["max_inclusive"] is None or delay_hours <= tier["max_inclusive"]
        if lower_ok and upper_ok:
            return tier
    return DELAY_COMPENSATION_TIERS[-1]  # unreachable: last tier is unbounded above


def compute_delay_compensation(booking: dict[str, Any]) -> PolicyResult:
    """
    Delay tiers, from the rule text:
        under 3 hours      -> ₹500 meal voucher
        more than 3 hours  -> meal voucher + lounge access
        more than 5 hours  -> meal voucher + hotel (DELAYED HOURS ONLY)

    BOUNDARY (the highest-risk silent bug in this project): the rule uses strict
    "more than", so 3.0h and 5.0h are gaps in the written policy. We resolve both
    DOWNWARD and record it in `assumptions`. `<=` vs `<` reads identically at a
    glance and the brief's own scenarios (4h, 6h) never probe the boundary, so
    this is tested explicitly in test_policy_and_escalation.py.
    """
    if not booking.get("airline_caused", False):
        return PolicyResult(
            disruption_type="delayed",
            customer_facts={"flight": booking.get("flight"), "delay_hours": booking.get("delay_hours")},
            notes=("Delay was not airline-caused; compensation policy does not apply.",),
        )

    delay_hours = float(booking["delay_hours"])
    tier = _select_delay_tier(delay_hours)

    facts: dict[str, Any] = {
        "flight": booking["flight"],
        "route": booking["route"],
        "scheduled_departure": booking["scheduled_departure"],
        "new_departure": booking["new_departure"],
        "delay_hours": delay_hours,
        "compensation_tier": tier["tier"],
        "compensation_description": tier["description"],
        "meal_voucher_value_rupees": MEAL_VOUCHER_VALUE_RUPEES,
    }
    notes = [f"Delay of {delay_hours:g}h qualifies for: {tier['description']}."]
    assumptions: list[str] = []

    if "hotel_accommodation_delayed_hours_only" in tier["entitlements"]:
        facts["hotel_hours_covered"] = delay_hours
        facts["hotel_coverage_scope"] = "delayed hours only — NOT a full night's stay"
        notes.append(
            f"Hotel accommodation covers the {delay_hours:g} delayed hours only. "
            "A full night's stay is beyond policy and cannot be approved by the agent."
        )
    if "lounge_access" in tier["entitlements"]:
        notes.append("Lounge access applies until the new departure time.")

    if delay_hours in (3.0, 5.0):
        assumptions.append(
            f"Delay is exactly {delay_hours:g}h. The rule says 'more than {delay_hours:g} hours' "
            "(strict), so this falls in a literal gap in the policy text; resolved to the "
            "LOWER tier to avoid granting compensation the policy does not state."
        )

    return PolicyResult(
        disruption_type="delayed",
        entitlements=tuple(tier["entitlements"]),
        customer_facts=facts,
        allowed_actions=(
            "Issue meal voucher and/or lounge access per the delay compensation rule",
            "Arrange hotel accommodation for the delayed-hours portion, where the delay qualifies",
        ),
        notes=tuple(notes),
        assumptions=tuple(assumptions),
    )


def resolve_unaffected(booking: dict[str, Any]) -> PolicyResult:
    """No disruption on this segment -> no disruption entitlements. Status info only."""
    return PolicyResult(
        disruption_type="unaffected",
        customer_facts={
            "flight": booking["flight"],
            "route": booking["route"],
            "date": booking["date"],
            "scheduled_departure": booking["scheduled_departure"],
            "status": "on schedule",
        },
        allowed_actions=("Provide the customer's own booking and flight status information",),
        notes=("This segment is unaffected; no disruption compensation is due on it.",),
    )


# --------------------------------------------------------------------------- #
# Rule family 3 — Fare Difference Rule
# --------------------------------------------------------------------------- #
def check_fare_difference(amount: float | None) -> dict[str, Any]:
    """
    "Agents cannot waive fare differences above ₹1,500 without supervisor approval."

    Strict `>` on the threshold: exactly ₹1,500 is waivable by the agent,
    ₹1,500.01 is not. Returns a plain dict (not PolicyResult) because this is a
    modifier layered onto a disruption outcome, not a disruption outcome itself.
    """
    if amount is None:
        return {"applicable": False, "amount_rupees": None, "requires_supervisor_approval": False}

    amount = float(amount)
    over = amount > FARE_DIFFERENCE_SUPERVISOR_THRESHOLD_RUPEES
    return {
        "applicable": True,
        "amount_rupees": amount,
        "threshold_rupees": FARE_DIFFERENCE_SUPERVISOR_THRESHOLD_RUPEES,
        "requires_supervisor_approval": over,
        "note": (
            f"Fare difference of ₹{amount:,.2f} exceeds the ₹"
            f"{FARE_DIFFERENCE_SUPERVISOR_THRESHOLD_RUPEES:,.0f} agent waiver limit; "
            "supervisor approval required."
            if over
            else f"Fare difference of ₹{amount:,.2f} is within the agent waiver limit."
        ),
    }


# --------------------------------------------------------------------------- #
# Rule family 4 — Loyalty Tier Rule
# --------------------------------------------------------------------------- #
def check_loyalty_priority(tier: str | None) -> bool:
    """
    "Gold and Platinum tier customers get priority rebooking (first access to
    next-available seats) but NO additional compensation beyond standard policy."

    The second clause is the one that matters for escalation: tier is never a
    reason to exceed policy amounts.
    """
    return bool(tier) and tier.strip().title() in PRIORITY_REBOOKING_TIERS


# --------------------------------------------------------------------------- #
# Composition — the single entry point every downstream layer uses.
# --------------------------------------------------------------------------- #
def build_policy_result(
    booking: dict[str, Any] | None,
    customer: dict[str, Any] | None,
    requested_fare_difference: float | None = None,
) -> PolicyResult:
    """
    Dispatch to the right rule family by booking status, then layer loyalty
    priority and any fare-difference modifier on top.

    Never let two layers compute policy independently — they drift. This is the
    only function that produces a PolicyResult for the graph.
    """
    if booking is None or customer is None:
        return PolicyResult(
            disruption_type="unknown",
            notes=("No booking identified; cannot determine entitlements.",),
        )

    status = booking.get("status")
    if status == "cancelled":
        result = resolve_cancellation(booking)
    elif status == "delayed":
        result = compute_delay_compensation(booking)
    else:
        result = resolve_unaffected(booking)

    priority = check_loyalty_priority(customer.get("loyalty_tier"))
    facts = dict(result.customer_facts)
    facts["customer_name"] = customer["name"]
    facts["loyalty_tier"] = customer["loyalty_tier"]
    facts["booking_reference"] = customer["booking_reference"]

    notes = list(result.notes)
    allowed = list(result.allowed_actions)
    if priority:
        notes.append(
            f"{customer['loyalty_tier']} tier: priority rebooking (first access to "
            "next-available seats). Tier grants NO compensation beyond standard policy."
        )
        allowed.append("Offer priority rebooking on next-available seats")

    fare = check_fare_difference(requested_fare_difference)
    if fare["applicable"]:
        facts["fare_difference"] = fare
        notes.append(fare["note"])

    return PolicyResult(
        disruption_type=result.disruption_type,
        entitlements=result.entitlements,
        customer_facts=facts,
        allowed_actions=tuple(allowed),
        notes=tuple(notes),
        priority_rebooking=priority,
        assumptions=result.assumptions,
    )

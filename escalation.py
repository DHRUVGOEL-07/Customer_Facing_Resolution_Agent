"""
Escalation gate — deterministic, zero LLM calls.

Each of the five "Prohibited (must escalate to a human agent)" bullets in the
Data Pack maps to exactly ONE private check function below. That 1:1 mapping is
deliberate: a reviewer can point at any bullet and at the function that enforces
it, instead of auditing one undocumented boolean expression.

INTENT TAXONOMY CONTRACT
------------------------
One label per DISTINCT POLICY OUTCOME — never one label per phrasing style.
If two labels can describe the same entitled request (e.g. `refund_request` and
some `cash_refund_extra`), the LLM will pick them inconsistently and a customer
who is simply OWED a refund gets escalated for nothing. The canonical set is
frozen in ALLOWED_INTENTS and the classifier prompt is generated from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from policy import PolicyResult, check_fare_difference

# --------------------------------------------------------------------------- #
# Canonical intent taxonomy. Single source of truth, shared with llm_nodes.py.
# --------------------------------------------------------------------------- #
ALLOWED_INTENTS: Final[dict[str, str]] = {
    # --- Entitled outcomes: granted by an existing rule. NEVER escalate alone. ---
    "cancellation_inquiry": "Asking about a cancelled flight / what happens next.",
    "rebooking_request": "Wants to be rebooked on another flight (airline-caused disruption).",
    "refund_request": "Wants their money back for an airline-caused cancellation. "
    "This is the outcome the Cancellation Rebooking Rule already grants.",
    "delay_inquiry": "Asking about a delayed flight / what they are owed.",
    "compensation_inquiry": "Asking what compensation the delay policy provides.",
    "hotel_request": "Asking for hotel accommodation for a qualifying delay.",
    "booking_status_inquiry": "Asking for their own booking or flight status.",
    # --- Outcomes NO rule grants. These are the escalation triggers. ---
    "compensation_beyond_policy": "Asking for more than the stated policy amounts "
    "(e.g. a FULL night's hotel when only delayed hours are covered, extra cash on top).",
    "class_upgrade": "Asking for a free/complimentary class upgrade. No rule grants this.",
    "voluntary_rebooking_higher_fare": "Wants to move to a higher-fare flight by choice "
    "(not airline-caused) — triggers the Fare Difference Rule.",
    "refund_to_different_method": "Wants the refund paid somewhere other than the "
    "original payment method.",
    "legal_threat": "Threatens legal action.",
    "formal_complaint": "States they are filing a formal complaint.",
    "non_airline_caused_exception": "Asking for an exception on something the airline "
    "did not cause (e.g. they missed the flight themselves).",
}

# Outcomes that no rule grants. `refund_request` is deliberately ABSENT — that is
# the taxonomy bug this set exists to avoid.
BEYOND_POLICY_ASKS: Final[frozenset[str]] = frozenset(
    {"compensation_beyond_policy", "class_upgrade"}
)

ALLOWED_SENTIMENTS: Final[tuple[str, ...]] = ("neutral", "confused", "frustrated", "angry")


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reasons: tuple[str, ...] = ()
    triggered_rules: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "No escalation required."


# --------------------------------------------------------------------------- #
# One check per Prohibited bullet. Each returns a reason string, or None.
# --------------------------------------------------------------------------- #
def _check_compensation_beyond_policy(
    intents: list[str], policy: PolicyResult | None
) -> str | None:
    """Bullet 1: "Approving any compensation beyond the stated policy amounts"."""
    hits = sorted(set(intents) & BEYOND_POLICY_ASKS)
    if not hits:
        return None
    detail = {
        "class_upgrade": "a complimentary class upgrade",
        "compensation_beyond_policy": "compensation beyond the stated policy amounts",
    }
    asked = " and ".join(detail[h] for h in hits)
    return f"Customer is requesting {asked}, which no service rule grants."


def _check_fare_waiver(amount: float | None) -> str | None:
    """Bullet 2: "Waiving a fare difference above ₹1,500"."""
    fare = check_fare_difference(amount)
    if fare["applicable"] and fare["requires_supervisor_approval"]:
        return (
            f"Fare difference of ₹{fare['amount_rupees']:,.2f} exceeds the ₹"
            f"{fare['threshold_rupees']:,.0f} agent waiver limit; supervisor approval required."
        )
    return None


def _check_non_airline_caused_exception(
    intents: list[str], policy: PolicyResult | None
) -> str | None:
    """Bullet 3: "Making exceptions for non-airline-caused disruptions"."""
    if "non_airline_caused_exception" in intents:
        return "Customer is requesting an exception for a disruption the airline did not cause."
    return None


def _check_legal_or_formal_complaint(intents: list[str], sentiment: str | None) -> str | None:
    """
    Bullet 4: "Handling threats of legal action or formal complaints —
    must be escalated immediately".

    Keyed on INTENT, never on sentiment. An angry customer is still a routine
    case; escalating on anger alone would push every frustrated-but-entitled
    customer to a human and defeat the agent's purpose.
    """
    if "legal_threat" in intents:
        return "Customer has threatened legal action — must be escalated immediately."
    if "formal_complaint" in intents:
        return "Customer is filing a formal complaint — must be escalated immediately."
    return None


def _check_refund_to_different_method(intents: list[str]) -> str | None:
    """Bullet 5: "Processing refunds to a different payment method than the original"."""
    if "refund_to_different_method" in intents:
        return (
            "Customer wants the refund issued to a different payment method; "
            "policy permits the original payment method only."
        )
    return None


# --------------------------------------------------------------------------- #
def check_escalation(
    intents: list[str] | None = None,
    sentiment: str | None = None,
    policy: PolicyResult | None = None,
    requested_fare_difference: float | None = None,
) -> EscalationDecision:
    """
    Collect every non-None check result. Escalate iff at least one fired.

    Tolerant of a missing/garbled classifier payload: `intents=None` degrades to
    "no prohibited intent detected" rather than raising inside a graph node.
    """
    intents = [i.strip() for i in (intents or []) if isinstance(i, str)]

    checks: tuple[tuple[str, str | None], ...] = (
        ("compensation_beyond_policy", _check_compensation_beyond_policy(intents, policy)),
        ("fare_waiver_above_threshold", _check_fare_waiver(requested_fare_difference)),
        ("non_airline_caused_exception", _check_non_airline_caused_exception(intents, policy)),
        ("legal_or_formal_complaint", _check_legal_or_formal_complaint(intents, sentiment)),
        ("refund_to_different_method", _check_refund_to_different_method(intents)),
    )

    fired = [(rule, reason) for rule, reason in checks if reason is not None]
    return EscalationDecision(
        escalate=bool(fired),
        reasons=tuple(reason for _, reason in fired),
        triggered_rules=tuple(rule for rule, _ in fired),
    )


def intent_catalogue_for_prompt() -> str:
    """Render ALLOWED_INTENTS for the classifier system prompt.

    Generated, not hand-copied: a hand-copied list drifts from the enforcement
    set the moment a label is added, which is exactly how the taxonomy bug
    reappears.
    """
    return "\n".join(f"- {name}: {desc}" for name, desc in ALLOWED_INTENTS.items())


def sanitize_intents(raw: Any) -> list[str]:
    """Drop anything the model hallucinated outside the canonical taxonomy."""
    if not isinstance(raw, list):
        return []
    return [i for i in raw if isinstance(i, str) and i.strip() in ALLOWED_INTENTS]

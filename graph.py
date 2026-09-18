"""
Graph wiring — LangGraph orchestration.

    identify_customer ─[needs clarification?]→ ask_for_booking_ref ─┐
                      └─→ extract_intent_node → run_policy_engine   │
                            → escalation_gate                        │
                                ├─[escalate]→ escalate_to_human ─────┤
                                └───────────→ compose_reply_node ────┤
                                                                      → log_action → END

Two invariants that prevent the classic bugs at this layer:

1. STATE SHAPE. `AgentState` is declared up front and no node returns a key that
   isn't in it. A typo'd key ("escalaton") in an untyped dict wouldn't raise
   until three nodes downstream, with a traceback pointing at the wrong node.

2. UPSTREAM-SKIPPED KEYS. The clarification branch bypasses intent, policy and
   escalation entirely, so every node uses `state.get(...)`, never `state[...]`.
   A KeyError deep in a graph is far harder to debug than a proactive None check.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

import llm_nodes
from data import find_customer, get_affected_segment
from escalation import EscalationDecision, check_escalation
from policy import PolicyResult, build_policy_result


def _append(left: list, right: list) -> list:
    """Reducer: action-log entries accumulate across nodes instead of overwriting."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    # inputs
    customer_message: str
    booking_ref: str | None
    # resolved
    customer: dict[str, Any] | None
    booking: dict[str, Any] | None
    needs_clarification: bool
    intents: list[str]
    sentiment: str
    requested_fare_difference: float | None
    policy: PolicyResult | None
    escalation: EscalationDecision | None
    # outputs
    reply: str
    action_log: Annotated[list[dict[str, Any]], _append]


def _entry(node: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"node": node, "detail": detail, **extra}


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def identify_customer(state: AgentState) -> dict[str, Any]:
    """Resolve PNR -> customer + affected segment. Falls back to scanning the
    message for a PNR when the caller didn't supply one (first turn in the UI)."""
    ref = state.get("booking_ref")
    if not ref:
        from data import CUSTOMERS

        message = (state.get("customer_message") or "").upper()
        ref = next((pnr for pnr in CUSTOMERS if pnr in message), None)

    customer = find_customer(ref)
    booking = get_affected_segment(ref) if customer else None

    if not customer:
        return {
            "booking_ref": None,
            "customer": None,
            "booking": None,
            "needs_clarification": True,
            "action_log": [_entry("identify_customer", "No valid booking reference found.")],
        }

    return {
        "booking_ref": customer["booking_reference"],
        "customer": customer,
        "booking": booking,
        "needs_clarification": False,
        "action_log": [
            _entry(
                "identify_customer",
                f"Identified {customer['name']} ({customer['loyalty_tier']}) on "
                f"{customer['booking_reference']} — {booking['status_detail'] if booking else 'no segment'}.",
            )
        ],
    }


def ask_for_booking_ref(state: AgentState) -> dict[str, Any]:
    """The 'ask only necessary questions' branch: one question, then stop."""
    return {
        "reply": (
            "Happy to help with that — I just need your booking reference "
            "(six characters, e.g. SK4821X) so I can pull up the right flight."
        ),
        "action_log": [_entry("ask_for_booking_ref", "Requested booking reference from customer.")],
    }


def extract_intent_node(state: AgentState) -> dict[str, Any]:
    extraction = llm_nodes.extract_intent(state.get("customer_message", ""))

    fare = extraction.requested_fare_difference_rupees
    # Fare differences are business data, not something the customer's phrasing
    # gets to set. If the model didn't find one and the case is a voluntary
    # higher-fare rebooking, leave it None so the reviewer sees an explicit
    # "amount unknown" rather than a number the model made up.
    return {
        "intents": extraction.intents,
        "sentiment": extraction.sentiment,
        "requested_fare_difference": fare,
        "action_log": [
            _entry(
                "extract_intent",
                f"Intents: {', '.join(extraction.intents) or 'none'} | "
                f"Sentiment: {extraction.sentiment}"
                + (f" | Fare difference cited: ₹{fare:,.2f}" if fare else ""),
            )
        ],
    }


def run_policy_engine(state: AgentState) -> dict[str, Any]:
    policy = build_policy_result(
        booking=state.get("booking"),
        customer=state.get("customer"),
        requested_fare_difference=state.get("requested_fare_difference"),
    )
    return {
        "policy": policy,
        "action_log": [
            _entry(
                "run_policy_engine",
                f"{policy.disruption_type} → entitlements: "
                f"{', '.join(policy.entitlements) or 'none'}"
                + (" | priority rebooking" if policy.priority_rebooking else ""),
            )
        ],
    }


def escalation_gate(state: AgentState) -> dict[str, Any]:
    decision = check_escalation(
        intents=state.get("intents"),
        sentiment=state.get("sentiment"),
        policy=state.get("policy"),
        requested_fare_difference=state.get("requested_fare_difference"),
    )
    return {
        "escalation": decision,
        "action_log": [_entry("escalation_gate", decision.summary, escalate=decision.escalate)],
    }


def escalate_to_human(state: AgentState) -> dict[str, Any]:
    """Escalation still composes a reply: the customer is told what IS granted
    deterministically, and only the out-of-policy part is handed onward."""
    policy = state.get("policy")
    escalation = state.get("escalation")
    reasons = escalation.reasons if escalation else ()

    reply = llm_nodes.compose_reply(
        customer_message=state.get("customer_message", ""),
        policy_facts=policy.as_facts() if policy else {},
        escalation_reasons=reasons,
        sentiment=state.get("sentiment", "neutral"),
    )
    return {
        "reply": reply,
        "action_log": [
            _entry("escalate_to_human", "Routed to human specialist: " + "; ".join(reasons))
        ],
    }


def compose_reply_node(state: AgentState) -> dict[str, Any]:
    policy = state.get("policy")
    reply = llm_nodes.compose_reply(
        customer_message=state.get("customer_message", ""),
        policy_facts=policy.as_facts() if policy else {},
        escalation_reasons=(),
        sentiment=state.get("sentiment", "neutral"),
    )
    return {"reply": reply, "action_log": [_entry("compose_reply", "Drafted customer reply.")]}


def log_action(state: AgentState) -> dict[str, Any]:
    """Terminal audit record — the 'preserve a clear conversation and action
    record' deliverable. `.get()` everywhere: the clarification branch skips
    intent/policy/escalation entirely."""
    escalation = state.get("escalation")
    return {
        "action_log": [
            _entry(
                "log_action",
                "Turn complete.",
                booking_ref=state.get("booking_ref"),
                escalated=bool(escalation.escalate) if escalation else False,
            )
        ]
    }


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def _route_identity(state: AgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "proceed"


def _route_escalation(state: AgentState) -> str:
    escalation = state.get("escalation")
    return "escalate" if escalation and escalation.escalate else "reply"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("identify_customer", identify_customer)
    g.add_node("ask_for_booking_ref", ask_for_booking_ref)
    g.add_node("extract_intent", extract_intent_node)
    g.add_node("run_policy_engine", run_policy_engine)
    g.add_node("escalation_gate", escalation_gate)
    g.add_node("escalate_to_human", escalate_to_human)
    g.add_node("compose_reply", compose_reply_node)
    g.add_node("log_action", log_action)

    g.add_edge(START, "identify_customer")
    g.add_conditional_edges(
        "identify_customer",
        _route_identity,
        {"clarify": "ask_for_booking_ref", "proceed": "extract_intent"},
    )
    g.add_edge("ask_for_booking_ref", "log_action")
    g.add_edge("extract_intent", "run_policy_engine")
    g.add_edge("run_policy_engine", "escalation_gate")
    g.add_conditional_edges(
        "escalation_gate",
        _route_escalation,
        {"escalate": "escalate_to_human", "reply": "compose_reply"},
    )
    g.add_edge("escalate_to_human", "log_action")
    g.add_edge("compose_reply", "log_action")
    g.add_edge("log_action", END)

    return g.compile()


def run_turn(customer_message: str, booking_ref: str | None = None) -> dict[str, Any]:
    """Single-turn convenience wrapper used by the Streamlit shell and the CLI."""
    return build_graph().invoke(
        {"customer_message": customer_message, "booking_ref": booking_ref, "action_log": []}
    )


if __name__ == "__main__":
    # Compile-time structural check — catches wiring mistakes before burning an API call.
    compiled = build_graph()
    print("Nodes:", sorted(compiled.get_graph().nodes.keys()))

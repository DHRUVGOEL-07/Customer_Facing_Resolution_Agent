"""
Streamlit shell — the clickable prototype (deliverable #1).

MULTI-TURN STATE is the thing that breaks here if you're careless. Streamlit
reruns the entire script on every interaction, so identity resolved on turn 1
must be carried forward explicitly in `session_state` or every message resets to
"who are you?" and the agent can never hold a conversation.

We carry forward ONLY the resolved-identity key (`booking_ref`). We deliberately
do NOT carry the whole previous state — a stale `reply` or `escalation` from
turn 1 leaking into turn 2's rendering is the other half of this bug.
"""

from __future__ import annotations

import streamlit as st

from data import CUSTOMERS, get_affected_segment
from graph import run_turn

st.set_page_config(page_title="Airline Resolution Agent", page_icon="✈️", layout="wide")

SCENARIOS = {
    "Scenario 1 — Priya Nair (Gold, SK4821X)": (
        "SK4821X",
        "My flight SK-204 to Goa got cancelled and nobody told me anything. I'm furious. "
        "I want a full cash refund, and a free upgrade to business class on my return "
        "flight for the trouble.",
    ),
    "Scenario 2 — Arvind Kulkarni (Silver, TR1190B)": (
        "TR1190B",
        "My flight SK-118 to Bengaluru is delayed 4 hours and I'm going to miss a "
        "connecting meeting. Since it's been such a long delay, I'd like hotel "
        "accommodation please.",
    ),
    "Scenario 3 — Meher Kaur (Platinum, WL7742)": (
        "WL7742",
        "SK-305 is delayed 6 hours. I want a full night's hotel stay, not just the "
        "delayed hours. I'd also rather be moved onto a different, higher-fare flight "
        "than wait — the fare difference is ₹2,000.",
    ),
}

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])      # [{"role","content"}]
st.session_state.setdefault("action_log", [])    # accumulated across turns
st.session_state.setdefault("booking_ref", None) # resolved identity, carried forward
st.session_state.setdefault("pending", None)     # scenario injected by a button


def reset() -> None:
    st.session_state.messages = []
    st.session_state.action_log = []
    st.session_state.booking_ref = None
    st.session_state.pending = None


# --------------------------------------------------------------------------- #
# Sidebar — identity, scenario launchers, and the live action record
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Session")
    ref = st.session_state.booking_ref
    if ref and ref in CUSTOMERS:
        customer = CUSTOMERS[ref]
        segment = get_affected_segment(ref)
        st.success(f"**{customer['name']}** · {customer['loyalty_tier']} · `{ref}`")
        if segment:
            st.caption(f"{segment['flight']} — {segment['route']}\n\n{segment['status_detail']}")
    else:
        st.info("No booking identified yet.")

    st.divider()
    st.subheader("Load a scenario")
    for label, (scenario_ref, text) in SCENARIOS.items():
        if st.button(label, use_container_width=True):
            reset()
            st.session_state.booking_ref = scenario_ref
            st.session_state.pending = text
            st.rerun()

    if st.button("Reset conversation", use_container_width=True):
        reset()
        st.rerun()

    st.divider()
    st.subheader("Action record")
    st.caption("Deliverable: 'preserve a clear conversation and action record'.")
    if not st.session_state.action_log:
        st.caption("_No actions yet._")
    for entry in st.session_state.action_log:
        icon = "🚩" if entry.get("escalate") or entry.get("escalated") else "•"
        st.markdown(f"{icon} **`{entry['node']}`** — {entry['detail']}")

# --------------------------------------------------------------------------- #
# Main pane
# --------------------------------------------------------------------------- #
st.title("✈️ Customer-Facing Resolution Agent")
st.caption(
    "All entitlements and escalation decisions are computed by a deterministic policy "
    "engine. The LLM classifies intent and phrases the reply — it never decides an amount."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type a customer message…") or st.session_state.pending
st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Resolving…"):
            try:
                result = run_turn(user_input, booking_ref=st.session_state.booking_ref)
            except Exception as exc:  # never leave the UI in a broken state
                st.error(f"The agent hit an error and could not complete this turn: {exc}")
                st.stop()

        reply = result.get("reply", "")
        escalation = result.get("escalation")
        if escalation is not None and escalation.escalate:
            st.warning("**Escalated to a human specialist**\n\n" + "\n".join(
                f"- {r}" for r in escalation.reasons
            ))
        st.markdown(reply)

        policy = result.get("policy")
        if policy is not None:
            with st.expander("Policy facts used to generate this reply"):
                st.json(policy.as_facts())
                if policy.assumptions:
                    st.caption("Assumptions: " + " ".join(policy.assumptions))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.action_log.extend(result.get("action_log", []))
    # Carry ONLY resolved identity forward — not the whole prior state.
    if result.get("booking_ref"):
        st.session_state.booking_ref = result["booking_ref"]
    st.rerun()

"""
Deterministic test suite. ZERO LLM calls, runs in well under a second.

Order is deliberate:
  1. the three scenarios from the brief (ground truth),
  2. boundary values (exactly 3h / 5h / ₹1,500),
  3. the taxonomy regression (a plain refund must not escalate).

Boundary tests written before scenario tests would mean perfecting edge cases
inside a design that might still be the wrong shape.
"""

from __future__ import annotations

import pytest

from data import CUSTOMERS, find_customer, get_affected_segment, get_segments
from escalation import BEYOND_POLICY_ASKS, check_escalation
from policy import (
    build_policy_result,
    check_fare_difference,
    check_loyalty_priority,
    compute_delay_compensation,
    resolve_cancellation,
)


# --------------------------------------------------------------------------- #
# 0. Data layer integrity
# --------------------------------------------------------------------------- #
def test_data_pack_transcribed_correctly():
    assert get_affected_segment("WL7742")["delay_hours"] == 6
    assert get_affected_segment("TR1190B")["delay_hours"] == 4
    assert get_affected_segment("SK4821X")["status"] == "cancelled"
    assert len(get_segments("SK4821X")) == 2  # return segment must not be lost
    assert CUSTOMERS["SK4821X"]["loyalty_tier"] == "Gold"
    assert find_customer("sk4821x") is CUSTOMERS["SK4821X"]  # case-insensitive
    assert find_customer("NOPE99") is None


# --------------------------------------------------------------------------- #
# 1. Scenarios from the brief
# --------------------------------------------------------------------------- #
def test_scenario_1_priya_cancelled_flight_entitlements():
    """Gold, SK4821X, SK-204 cancelled: rebooking OR refund, plus priority rebooking."""
    policy = build_policy_result(get_affected_segment("SK4821X"), find_customer("SK4821X"))
    assert policy.disruption_type == "cancelled"
    assert set(policy.entitlements) == {"free_rebooking_within_24h", "full_refund"}
    assert policy.priority_rebooking is True
    assert policy.customer_facts["refund_processing_business_days"] == 7


def test_scenario_1_priya_upgrade_request_escalates():
    """The upgrade ask — not the refund ask — is what escalates this case."""
    policy = build_policy_result(get_affected_segment("SK4821X"), find_customer("SK4821X"))
    decision = check_escalation(
        intents=["cancellation_inquiry", "refund_request", "class_upgrade"],
        sentiment="angry",
        policy=policy,
    )
    assert decision.escalate is True
    assert decision.triggered_rules == ("compensation_beyond_policy",)
    assert "upgrade" in decision.summary.lower()


def test_scenario_2_arvind_4h_delay_gets_voucher_and_lounge_no_hotel():
    """Silver, 4h delay: over_3h tier. He ASKS for a hotel; 4h does not qualify."""
    policy = build_policy_result(get_affected_segment("TR1190B"), find_customer("TR1190B"))
    assert policy.customer_facts["compensation_tier"] == "over_3h"
    assert set(policy.entitlements) == {"meal_voucher", "lounge_access"}
    assert "hotel_accommodation_delayed_hours_only" not in policy.entitlements
    assert policy.priority_rebooking is False  # Silver gets no priority rebooking

    # Frustration alone must NOT escalate a fully in-policy case.
    assert check_escalation(
        intents=["delay_inquiry", "compensation_inquiry"], sentiment="frustrated", policy=policy
    ).escalate is False


def test_scenario_3_meher_6h_delay_hotel_is_delayed_hours_only():
    """Platinum, 6h delay: over_5h tier, hotel capped at the delayed hours."""
    policy = build_policy_result(get_affected_segment("WL7742"), find_customer("WL7742"))
    assert policy.customer_facts["compensation_tier"] == "over_5h"
    assert "hotel_accommodation_delayed_hours_only" in policy.entitlements
    assert policy.customer_facts["hotel_hours_covered"] == 6.0
    assert policy.priority_rebooking is True


def test_scenario_3_meher_full_night_and_2000_fare_both_escalate():
    """Two independent prohibited asks must both surface, not just the first."""
    policy = build_policy_result(
        get_affected_segment("WL7742"), find_customer("WL7742"), requested_fare_difference=2000.0
    )
    decision = check_escalation(
        intents=["delay_inquiry", "compensation_beyond_policy", "voluntary_rebooking_higher_fare"],
        sentiment="neutral",
        policy=policy,
        requested_fare_difference=2000.0,
    )
    assert decision.escalate is True
    assert set(decision.triggered_rules) == {
        "compensation_beyond_policy",
        "fare_waiver_above_threshold",
    }
    assert len(decision.reasons) == 2


# --------------------------------------------------------------------------- #
# 2. Boundary values — the silent-bug zone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "delay_hours, expected_tier",
    [
        (0.5, "under_3h"),
        (2.99, "under_3h"),
        (3.0, "under_3h"),   # "more than 3 hours" is strict -> resolves DOWN
        (3.01, "over_3h"),
        (5.0, "over_3h"),    # "more than 5 hours" is strict -> resolves DOWN
        (5.01, "over_5h"),
        (12.0, "over_5h"),
    ],
)
def test_delay_tier_boundaries_are_strict(delay_hours, expected_tier):
    booking = {
        "flight": "SK-TEST", "route": "A → B", "scheduled_departure": "10:00",
        "new_departure": "12:00", "delay_hours": delay_hours, "airline_caused": True,
    }
    result = compute_delay_compensation(booking)
    assert result.customer_facts["compensation_tier"] == expected_tier
    if delay_hours in (3.0, 5.0):
        assert result.assumptions, "exact-boundary resolution must be recorded as an assumption"


def test_fare_difference_threshold_is_strict():
    assert check_fare_difference(1499.99)["requires_supervisor_approval"] is False
    assert check_fare_difference(1500.00)["requires_supervisor_approval"] is False  # at limit: fine
    assert check_fare_difference(1500.01)["requires_supervisor_approval"] is True   # one paisa over
    assert check_fare_difference(None)["applicable"] is False


def test_loyalty_priority_tiers():
    assert check_loyalty_priority("Gold") is True
    assert check_loyalty_priority("Platinum") is True
    assert check_loyalty_priority("Silver") is False
    assert check_loyalty_priority(None) is False


def test_non_airline_caused_grants_nothing():
    booking = {"flight": "SK-X", "route": "A → B", "status_detail": "Cancelled (weather/customer)",
               "scheduled_departure": "10:00", "airline_caused": False, "delay_hours": 0}
    assert resolve_cancellation(booking).entitlements == ()


# --------------------------------------------------------------------------- #
# 3. Taxonomy regression — the bug that escalates entitled customers
# --------------------------------------------------------------------------- #
def test_plain_refund_request_alone_does_not_escalate():
    """
    A customer owed a refund under the Cancellation Rebooking Rule must be served
    by the agent, not bounced to a human. `refund_request` is an ENTITLED outcome
    and must never appear in the beyond-policy set.
    """
    policy = build_policy_result(get_affected_segment("SK4821X"), find_customer("SK4821X"))
    decision = check_escalation(
        intents=["cancellation_inquiry", "refund_request"], sentiment="angry", policy=policy
    )
    assert decision.escalate is False
    assert decision.reasons == ()


def test_refund_request_is_not_in_beyond_policy_set():
    """Guards the taxonomy contract itself, not just one call site."""
    assert "refund_request" not in BEYOND_POLICY_ASKS
    assert BEYOND_POLICY_ASKS == {"compensation_beyond_policy", "class_upgrade"}


def test_anger_alone_never_escalates():
    """Escalation keys on intent, never on sentiment."""
    for sentiment in ("neutral", "confused", "frustrated", "angry"):
        assert check_escalation(intents=["delay_inquiry"], sentiment=sentiment).escalate is False


def test_legal_threat_and_formal_complaint_escalate_immediately():
    assert check_escalation(intents=["legal_threat"]).escalate is True
    assert check_escalation(intents=["formal_complaint"]).escalate is True
    assert check_escalation(intents=["refund_to_different_method"]).escalate is True
    assert check_escalation(intents=["non_airline_caused_exception"]).escalate is True


def test_escalation_gate_survives_garbage_classifier_output():
    """A failed/hallucinating LLM must degrade to 'no prohibited intent', not crash."""
    assert check_escalation(intents=None).escalate is False
    assert check_escalation(intents=["not_a_real_label", None, 42]).escalate is False  # type: ignore[list-item]

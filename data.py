"""
Data layer — Assignment 3: Customer-Facing Resolution Agent (Airline Disruption).

SINGLE SOURCE OF TRUTH. Everything here is transcribed verbatim from the supplied
Data Pack. No invented customers, flights, or policy values.

Design rule enforced in this module:
    Rule CONSTANTS live here. Rule LOGIC lives in policy.py.
Conflating the two makes the policy engine untestable in isolation.
"""

from __future__ import annotations

from typing import Any, Final

# --------------------------------------------------------------------------- #
# Exercise clock (Data Pack: "This exercise is set on Wednesday, 23 September 2026.")
# --------------------------------------------------------------------------- #
EXERCISE_DATE: Final[str] = "2026-09-23"

# --------------------------------------------------------------------------- #
# 1. Customer Profiles
# --------------------------------------------------------------------------- #
CUSTOMERS: Final[dict[str, dict[str, Any]]] = {
    "SK4821X": {
        "name": "Priya Nair",
        "loyalty_tier": "Gold",
        "booking_reference": "SK4821X",
        "email": "priya.nair@example.com",
        "phone": "+91-98xxxxxxx1",
        "flights_last_12_months": 6,
        "prior_complaints": 1,
        "prior_complaint_detail": "delayed baggage, resolved with voucher",
    },
    "TR1190B": {
        "name": "Arvind Kulkarni",
        "loyalty_tier": "Silver",
        "booking_reference": "TR1190B",
        "email": "arvind.kulkarni@example.com",
        "phone": "+91-98xxxxxxx2",
        "flights_last_12_months": 3,
        "prior_complaints": 0,
        "prior_complaint_detail": None,
    },
    "WL7742": {
        "name": "Meher Kaur",
        "loyalty_tier": "Platinum",
        "booking_reference": "WL7742",
        "email": "meher.kaur@example.com",
        "phone": "+91-98xxxxxxx3",
        "flights_last_12_months": 10,
        "prior_complaints": 1,
        "prior_complaint_detail": "overbooking, resolved with a tier-status upgrade",
    },
}

# --------------------------------------------------------------------------- #
# 2. Booking / Transaction Data
#
# NOTE: PNR SK4821X covers TWO segments (outbound + return). The store is
# therefore PNR -> list[segment]; `get_affected_segment()` picks the disrupted
# one. Modelling this as a flat PNR->dict would silently lose Priya's return
# flight, which Scenario 1 explicitly asks about ("upgrade on her return flight").
# --------------------------------------------------------------------------- #
BOOKINGS: Final[dict[str, list[dict[str, Any]]]] = {
    "SK4821X": [
        {
            "pnr": "SK4821X",
            "segment_id": "SK4821X-OUT",
            "flight": "SK-204",
            "route": "Delhi → Goa",
            "date": "2026-09-23",
            "scheduled_departure": "18:40",
            "new_departure": None,
            "status": "cancelled",
            "status_detail": "Cancelled (operational reasons)",
            "airline_caused": True,
            "delay_hours": 0.0,
        },
        {
            "pnr": "SK4821X",
            "segment_id": "SK4821X-RET",
            "flight": "Return",
            "route": "Goa → Delhi",
            "date": "2026-09-25",
            "scheduled_departure": "16:20",
            "new_departure": None,
            "status": "unaffected",
            "status_detail": "Unaffected",
            "airline_caused": False,
            "delay_hours": 0.0,
        },
    ],
    "TR1190B": [
        {
            "pnr": "TR1190B",
            "segment_id": "TR1190B-OUT",
            "flight": "SK-118",
            "route": "Mumbai → Bengaluru",
            "date": "2026-09-23",
            "scheduled_departure": "07:10",
            "new_departure": "11:10",
            "status": "delayed",
            "status_detail": "Delayed 4h (new departure 11:10)",
            "airline_caused": True,
            "delay_hours": 4.0,
        }
    ],
    "WL7742": [
        {
            "pnr": "WL7742",
            "segment_id": "WL7742-OUT",
            "flight": "SK-305",
            "route": "Delhi → Hyderabad",
            "date": "2026-09-23",
            "scheduled_departure": "14:00",
            "new_departure": "20:00",
            "status": "delayed",
            "status_detail": "Delayed 6h (new departure 20:00)",
            "airline_caused": True,
            "delay_hours": 6.0,
        }
    ],
}

# --------------------------------------------------------------------------- #
# 3. Service Rule CONSTANTS  (logic lives in policy.py)
# --------------------------------------------------------------------------- #

# Cancellation Rebooking Rule
CANCELLATION_REBOOKING_WINDOW_HOURS: Final[int] = 24

# Refund Processing Rule
REFUND_PROCESSING_BUSINESS_DAYS: Final[int] = 7
REFUND_TO_ORIGINAL_METHOD_ONLY: Final[bool] = True

# Delay Compensation Rule.
# Rule text is: "under 3 hours" / "more than 3 hours" / "more than 5 hours".
# Both upper boundaries are STRICT inequalities in the source text, so the
# exact values 3.0 and 5.0 are literal gaps in the written policy. We resolve
# both gaps DOWNWARD (the conservative reading — never grant more than the
# policy provably states) and surface it as an assumption, rather than silently
# picking `>=` and over-compensating. See policy.compute_delay_compensation.
DELAY_COMPENSATION_TIERS: Final[tuple[dict[str, Any], ...]] = (
    {
        "tier": "under_3h",
        "min_exclusive": None,
        "max_inclusive": 3.0,
        "entitlements": ("meal_voucher",),
        "description": "₹500 meal voucher",
    },
    {
        "tier": "over_3h",
        "min_exclusive": 3.0,
        "max_inclusive": 5.0,
        "entitlements": ("meal_voucher", "lounge_access"),
        "description": "₹500 meal voucher + lounge access",
    },
    {
        "tier": "over_5h",
        "min_exclusive": 5.0,
        "max_inclusive": None,
        "entitlements": ("meal_voucher", "hotel_accommodation_delayed_hours_only"),
        "description": (
            "₹500 meal voucher + hotel accommodation covering only the delayed "
            "hours (not a full night's stay)"
        ),
    },
)
MEAL_VOUCHER_VALUE_RUPEES: Final[int] = 500

# Fare Difference Rule
FARE_DIFFERENCE_SUPERVISOR_THRESHOLD_RUPEES: Final[float] = 1500.0

# Loyalty Tier Rule
PRIORITY_REBOOKING_TIERS: Final[frozenset[str]] = frozenset({"Gold", "Platinum"})

# --------------------------------------------------------------------------- #
# 4. Allowed vs Prohibited (verbatim, used to label escalation reasons)
# --------------------------------------------------------------------------- #
PROHIBITED_ACTIONS: Final[tuple[str, ...]] = (
    "Approving any compensation beyond the stated policy amounts",
    "Waiving a fare difference above ₹1,500",
    "Making exceptions for non-airline-caused disruptions (e.g., customer missed the flight)",
    "Handling threats of legal action or formal complaints — must be escalated immediately",
    "Processing refunds to a different payment method than the original",
)


# --------------------------------------------------------------------------- #
# Lookup helpers — the ONLY sanctioned way to read the store.
# --------------------------------------------------------------------------- #
def find_customer(booking_ref: str | None) -> dict[str, Any] | None:
    """Case-insensitive PNR lookup. Returns None for unknown/missing refs."""
    if not booking_ref:
        return None
    return CUSTOMERS.get(booking_ref.strip().upper())


def get_segments(booking_ref: str | None) -> list[dict[str, Any]]:
    """All segments on a PNR. Empty list for unknown refs (never raises)."""
    if not booking_ref:
        return []
    return BOOKINGS.get(booking_ref.strip().upper(), [])


def get_affected_segment(booking_ref: str | None) -> dict[str, Any] | None:
    """
    The segment the customer is contacting us about: the disrupted one.
    Priority: cancelled > delayed > first segment. Returns None if PNR unknown.
    """
    segments = get_segments(booking_ref)
    if not segments:
        return None
    for status in ("cancelled", "delayed"):
        for seg in segments:
            if seg["status"] == status:
                return seg
    return segments[0]

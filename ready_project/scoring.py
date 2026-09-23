"""Transparent baseline for already eligible contractors; no quality claims."""

from decimal import Decimal
import math

RANKING_METHOD = "initial_price_then_id"
SCORING_VERSION = "baseline-price-v1"


def numeric_value(value, field):
    """Reject malformed normalized data instead of silently changing ranking."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be a finite number")
    number = Decimal(str(value))
    if not number.is_finite() or not math.isfinite(float(number)):
        raise ValueError(f"{field} must be a finite number")
    return number


def score_candidate(candidate):
    price = numeric_value(candidate["price_from_kzt"], "price_from_kzt")
    if price < 0:
        raise ValueError("price_from_kzt must be non-negative")
    score = round(-float(price), 6)
    return score, {"initial_price": score}

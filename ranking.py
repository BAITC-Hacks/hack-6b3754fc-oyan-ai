"""Stable ranking of the complete list supplied by the eligibility core."""

from copy import deepcopy

from explanations import collect_evidence
from scoring import RANKING_METHOD, SCORING_VERSION, numeric_value, score_candidate


def rank_candidates(candidates, query):
    """Return independent profile copies sorted by score, price, string ID.

    Prices must be finite non-negative numbers; IDs must be unique nonempty
    strings or integers after str conversion. Invalid data raises ValueError.
    The core owns query validation, eligibility and slicing to TOP-3.
    """
    ranked = []
    seen = set()
    for candidate in candidates:
        identifier = candidate.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError("id must be a nonempty string or integer")
        canonical_id = str(identifier)
        if canonical_id in seen:
            raise ValueError(f"duplicate canonical id: {canonical_id}")
        seen.add(canonical_id)
        row = deepcopy(candidate)
        row["score"], row["score_breakdown"] = score_candidate(row)
        row["evidence"] = collect_evidence(row, query)
        ranked.append(row)
    ranked.sort(key=lambda row: (-row["score"], numeric_value(row["price_from_kzt"], "price_from_kzt"), str(row["id"])))
    return ranked

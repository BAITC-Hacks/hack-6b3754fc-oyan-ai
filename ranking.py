"""Stable ranking of the complete list supplied by the eligibility core."""

from copy import deepcopy
import os

from explanations import collect_evidence
from scoring import RANKING_METHOD, SCORING_VERSION, numeric_value, score_candidate
from semantic import SemanticCache, SemanticCacheError

# Frozen at import/startup: changing environment or the file cannot silently
# change ranking in a live process. No automatic downgrade on a missing cache.
_MODE = os.environ.get("RANKING_MODE", "baseline")
_SEMANTIC_CACHE = None
if _MODE == "semantic":
    _SEMANTIC_CACHE = SemanticCache.from_path(
        os.environ.get("RANKING_SEMANTIC_CACHE", ""),
        os.environ.get("RANKING_SEMANTIC_SHA256", ""),
    )
    RANKING_METHOD = "cached_sentence_embeddings_cosine"
    SCORING_VERSION = f"semantic-cosine-v1:{_SEMANTIC_CACHE.sha256}"
elif _MODE != "baseline":
    raise SemanticCacheError("RANKING_MODE must be baseline or semantic")


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
        row.pop("evidence", None)  # Recompute derived facts, including on re-ranking.
        row["evidence"] = collect_evidence(row, query)
        if _SEMANTIC_CACHE is not None:
            description = row.get("description", "")
            row["score"] = _SEMANTIC_CACHE.similarity(description, query)
            row["score_breakdown"] = {"semantic_similarity": row["score"]}
            row["evidence"] = [fact for fact in row["evidence"] if fact["field"] != "description"]
            excerpt = _SEMANTIC_CACHE.excerpt(description, query)
            if excerpt:
                row["evidence"].append(excerpt)
            row["evidence"].append(_SEMANTIC_CACHE.evidence(description, query, row["score"]))
        ranked.append(row)
    ranked.sort(key=lambda row: (-row["score"], numeric_value(row["price_from_kzt"], "price_from_kzt"), str(row["id"])))
    return ranked

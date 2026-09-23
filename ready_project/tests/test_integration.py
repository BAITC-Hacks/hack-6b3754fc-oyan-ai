from __future__ import annotations

from pathlib import Path

from data_loader import load_contractors
from recommender import recommend


DATASET = Path("docs/hackathon dataset anonymized .csv")


def deterministic_test_ranker(candidates: list[dict], _query: dict) -> list[dict]:
    """Test-only adapter; production success still requires Participant 2."""

    return [
        {
            **candidate,
            "score": 0.0,
            "score_breakdown": {},
            "evidence": [{"field": "id", "value": candidate["id"]}],
        }
        for candidate in sorted(candidates, key=lambda item: item["id"])
    ]


deterministic_test_ranker.ranking_method = "test-only canonical id order"
deterministic_test_ranker.scoring_version = "test-v1"


def build_test_cards(candidates: list[dict], query: dict) -> list[dict]:
    return [
        {
            **candidate,
            "category": query["category"],
            "explanation": (
                f"Начальная цена от {candidate['price_from_kzt']} ₸ не выше бюджета. "
                f"По календарю нет отметки о занятости на {query['date']}."
            ),
        }
        for candidate in candidates
    ]


def real_query(*, budget: int, event_type: str) -> dict:
    return {
        "city": "Алматы",
        "date": "2026-10-10",
        "event_type": event_type,
        "category": "Ведущий",
        "budget": budget,
        "duration": 6,
        "language": "русский",
    }


def test_real_csv_positive_pipeline_and_top_three_boundary() -> None:
    contractors = load_contractors(DATASET)

    response = recommend(
        real_query(budget=2_000_000, event_type="корпоратив"),
        contractors,
        ranker=deterministic_test_ranker,
        card_builder=build_test_cards,
    )

    assert response["status"] == "success"
    assert response["stats"]["city_category_total"] == 10
    assert response["stats"]["eligible_count"] == 6
    assert response["stats"]["returned_count"] == 3
    assert response["stats"]["rejected_first_reason"]["busy"] == 4
    assert len(response["results"]) == 3


def test_real_csv_empty_pipeline_explains_primary_reasons() -> None:
    contractors = load_contractors(DATASET)

    response = recommend(
        real_query(budget=500_000, event_type="свадьба"), contractors
    )

    assert response["status"] == "no_matching_candidates"
    assert response["stats"]["city_category_total"] == 10
    assert response["stats"]["eligible_count"] == 0
    assert response["stats"]["rejected_first_reason"] == {
        "busy": 4,
        "over_budget": 5,
        "wrong_format": 1,
        "wrong_language": 0,
        "too_short": 0,
    }
    assert "ни один не прошёл" in response["message"]

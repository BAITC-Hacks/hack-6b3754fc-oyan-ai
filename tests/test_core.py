from __future__ import annotations

from copy import deepcopy

import pytest

from contracts import IntegrationContractError, IntegrationNotReadyError, QueryValidationError
from recommender import recommend


def profile(
    identifier: str,
    *,
    category: str = "Ведущий",
    city: str = "Алматы",
    price: int = 100_000,
    formats: list[str] | None = None,
    languages: list[str] | None = None,
    max_hours: int | None = 6,
    busy_dates: list[str] | None = None,
) -> dict:
    return {
        "id": identifier,
        "anon_name": f"Профиль {identifier}",
        "categories": [category],
        "city": city,
        "city_imputed": False,
        "synthetic": True,
        "price_from_kzt": price,
        "price_imputed": False,
        "event_formats": formats or ["свадьба"],
        "languages": languages or ["русский"],
        "max_hours": max_hours,
        "busy_dates": busy_dates or [],
        "description": f"Описание профиля {identifier} для теста.",
    }


def query(**overrides: object) -> dict:
    result = {
        "city": "Алматы",
        "date": "2026-10-10",
        "event_type": "свадьба",
        "category": "Ведущий",
        "budget": 100_000,
        "duration": 6,
        "language": "русский",
    }
    result.update(overrides)
    return result


def ranker(candidates: list[dict], _query: dict) -> list[dict]:
    return [
        {
            **candidate,
            "score": 1.0,
            "score_breakdown": {"test": 1.0},
            "evidence": [{"field": "id", "value": candidate["id"]}],
        }
        for candidate in sorted(candidates, key=lambda item: item["id"], reverse=True)
    ]


ranker.ranking_method = "deterministic test adapter"
ranker.scoring_version = "test-v1"


def card_builder(candidates: list[dict], request: dict) -> list[dict]:
    return [
        {
            **candidate,
            "category": request["category"],
            "explanation": (
                f"Начальная цена от {candidate['price_from_kzt']} ₸ не выше бюджета. "
                f"В профиле указан идентификатор {candidate['id']}."
            ),
        }
        for candidate in candidates
    ]


def run(request: dict, contractors: list[dict]) -> dict:
    return recommend(request, contractors, ranker=ranker, card_builder=card_builder)


def test_category_not_found_does_not_call_ranking() -> None:
    response = recommend(query(category="Фотограф"), [profile("A")])

    assert response["status"] == "category_not_found"
    assert response["results"] == []
    assert response["stats"]["city_category_total"] == 0


def test_all_rejected_preserves_all_reasons_but_counts_first_only() -> None:
    contractors = [
        profile(
            "A",
            price=200_000,
            formats=["корпоратив"],
            busy_dates=["2026-10-10"],
        ),
        profile("B", price=200_000),
        profile("C", formats=["корпоратив"]),
    ]

    response = recommend(query(), contractors)

    assert response["status"] == "no_matching_candidates"
    assert response["stats"]["rejected_first_reason"] == {
        "busy": 1,
        "over_budget": 1,
        "wrong_format": 1,
        "wrong_language": 0,
        "too_short": 0,
    }
    assert response["rejections"][0]["reasons"] == [
        "busy",
        "over_budget",
        "wrong_format",
    ]
    assert response["stats"]["city_category_total"] == 3


@pytest.mark.parametrize(
    ("candidate", "request_data"),
    [
        (profile("budget", price=100_000), query(budget=100_000)),
        (profile("duration", max_hours=6), query(duration=6)),
        (profile("unbound", max_hours=None), query(duration=24)),
        (profile("optional-language", languages=["русский"]), query(language=None)),
    ],
)
def test_eligibility_boundaries_pass(candidate: dict, request_data: dict) -> None:
    response = run(request_data, [candidate])

    assert response["status"] == "success"
    assert response["stats"]["eligible_count"] == 1


def test_busy_venue_is_rejected() -> None:
    venue = profile(
        "venue",
        category="Банкетный зал",
        max_hours=None,
        busy_dates=["2026-10-10"],
    )

    response = recommend(query(category="Банкетный зал"), [venue])

    assert response["status"] == "no_matching_candidates"
    assert response["rejections"][0]["primary_reason"] == "busy"


def test_optional_language_and_duration_filters() -> None:
    contractor = profile("A", languages=["русский"], max_hours=4)

    response = recommend(query(language="казахский", duration=6), [contractor])

    assert response["status"] == "no_matching_candidates"
    assert response["rejections"][0]["reasons"] == ["wrong_language", "too_short"]


def test_success_passes_all_eligible_to_ranker_then_slices_three() -> None:
    received: list[str] = []

    def tracking_ranker(candidates: list[dict], request: dict) -> list[dict]:
        received.extend(candidate["id"] for candidate in candidates)
        return ranker(candidates, request)

    tracking_ranker.ranking_method = "tracking adapter"
    tracking_ranker.scoring_version = "test-v1"
    contractors = [profile(identifier) for identifier in ["A", "B", "C", "D"]]

    response = recommend(
        query(), contractors, ranker=tracking_ranker, card_builder=card_builder
    )

    assert set(received) == {"A", "B", "C", "D"}
    assert [item["id"] for item in response["results"]] == ["D", "C", "B"]
    assert response["stats"]["eligible_count"] == 4
    assert response["stats"]["returned_count"] == 3
    assert response["message"] == "Условия проходят 4, показываем 3."


def test_one_or_two_results_have_honest_message() -> None:
    response = run(query(), [profile("A"), profile("B")])

    assert len(response["results"]) == 2
    assert "всего 2" in response["message"]


def test_missing_participant_two_module_is_an_integration_error(monkeypatch) -> None:
    def missing_ranking(name: str):
        assert name == "ranking"
        raise ModuleNotFoundError("Simulated missing ranking module", name="ranking")

    monkeypatch.setattr("recommender.importlib.import_module", missing_ranking)
    with pytest.raises(IntegrationNotReadyError, match="Participant 2"):
        recommend(query(), [profile("A")])


def test_ranker_must_return_every_eligible_candidate_once() -> None:
    def broken_ranker(candidates: list[dict], _request: dict) -> list[dict]:
        return candidates[:1]

    with pytest.raises(IntegrationContractError, match="every eligible candidate"):
        recommend(
            query(),
            [profile("A"), profile("B")],
            ranker=broken_ranker,
            card_builder=card_builder,
        )


def test_inputs_are_not_mutated() -> None:
    request = query()
    contractors = [profile("A"), profile("B")]
    request_before = deepcopy(request)
    contractors_before = deepcopy(contractors)

    run(request, contractors)

    assert request == request_before
    assert contractors == contractors_before


@pytest.mark.parametrize(
    "invalid_date",
    ["2026-09-22", "2027-01-01", "2026-9-23", "not-a-date"],
)
def test_invalid_or_out_of_window_date_is_input_error(invalid_date: str) -> None:
    with pytest.raises(QueryValidationError, match="date"):
        recommend(query(date=invalid_date), [])


@pytest.mark.parametrize("duration", [float("inf"), float("-inf"), float("nan"), 0, -1, True])
def test_invalid_duration_is_input_error(duration: object) -> None:
    with pytest.raises(QueryValidationError, match="duration"):
        recommend(query(duration=duration), [profile("A")])


@pytest.mark.parametrize("duration", [None, 0.5, 6, 6.0])
def test_finite_and_optional_duration_remain_supported(duration: object) -> None:
    assert run(query(duration=duration), [profile("A")])["status"] == "success"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("price_from_kzt", 999_999), ("busy_dates", ["2026-10-10"]),
     ("languages", ["английский"]), ("description", "Changed description")],
)
@pytest.mark.parametrize("stage", ["ranking", "cards"])
def test_integrations_cannot_change_source_fields(field, replacement, stage) -> None:
    def changing_ranker(candidates, request):
        ranked = ranker(candidates, request)
        ranked[0][field] = replacement
        return ranked

    def changing_builder(candidates, request):
        candidates[0][field] = replacement
        return card_builder(candidates, request)

    with pytest.raises(IntegrationContractError, match="changed source field"):
        recommend(
            query(), [profile("A")],
            ranker=changing_ranker if stage == "ranking" else ranker,
            card_builder=changing_builder if stage == "cards" else card_builder,
        )


@pytest.mark.parametrize("mutation", ["clear", "pop", "reverse"])
def test_builder_cannot_change_finalist_count_or_order_in_place(mutation) -> None:
    def changing_builder(candidates, request):
        getattr(candidates, mutation)()
        return card_builder(candidates, request)

    with pytest.raises(IntegrationContractError, match="finalist"):
        recommend(
            query(), [profile("A"), profile("B")],
            ranker=ranker, card_builder=changing_builder,
        )


def test_ranker_malformed_id_is_an_integration_error() -> None:
    def broken_ranker(candidates, request):
        ranked = ranker(candidates, request)
        ranked[0]["id"] = ["A"]
        return ranked

    with pytest.raises(IntegrationContractError, match="string ids"):
        recommend(query(), [profile("A")], ranker=broken_ranker, card_builder=card_builder)

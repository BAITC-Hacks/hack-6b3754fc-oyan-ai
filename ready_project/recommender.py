"""Eligibility filtering and integration shell for contractor recommendations."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from contracts import (
    Contractor,
    DatasetLoadError,
    IntegrationContractError,
    IntegrationNotReadyError,
    QueryValidationError,
    REJECTION_REASONS,
    RecommendationQuery,
)

Ranker = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]
CardBuilder = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _first_validation_error(error: ValidationError) -> str:
    problem = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in problem["loc"])
    return f"field {location}: {problem['msg']}"


def validate_query(query: dict[str, Any]) -> dict[str, Any]:
    """Validate the public query and return a detached JSON-compatible copy."""

    if not isinstance(query, dict):
        raise QueryValidationError("Query must be a dictionary")
    try:
        validated = RecommendationQuery.model_validate(query)
    except ValidationError as error:
        raise QueryValidationError(f"Invalid query: {_first_validation_error(error)}") from error
    return validated.model_dump(mode="json")


def _validated_contractors(contractors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(contractors, list):
        raise DatasetLoadError("Contractors must be a list")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_contractor in enumerate(contractors):
        try:
            contractor = Contractor.model_validate(raw_contractor)
        except ValidationError as error:
            raise DatasetLoadError(
                f"Invalid contractor at index {index}: {_first_validation_error(error)}"
            ) from error
        if contractor.id in seen_ids:
            raise DatasetLoadError(
                f"Duplicate contractor id {contractor.id!r} at index {index}"
            )
        seen_ids.add(contractor.id)
        validated.append(contractor.model_dump(mode="json"))
    return validated


def evaluate_candidate(candidate: dict[str, Any], query: dict[str, Any]) -> list[str]:
    """Return all rejection reasons in the fixed business priority order."""

    reasons: list[str] = []
    if query["date"] in candidate["busy_dates"]:
        reasons.append("busy")
    if candidate["price_from_kzt"] > query["budget"]:
        reasons.append("over_budget")
    if _normalize(query["event_type"]) not in {
        _normalize(value) for value in candidate["event_formats"]
    }:
        reasons.append("wrong_format")
    if query["language"] is not None and _normalize(query["language"]) not in {
        _normalize(value) for value in candidate["languages"]
    }:
        reasons.append("wrong_language")
    if (
        query["duration"] is not None
        and candidate["max_hours"] is not None
        and query["duration"] > candidate["max_hours"]
    ):
        reasons.append("too_short")
    return reasons


def filter_candidates(
    query: dict[str, Any], contractors: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], int]:
    """Select the city/category group and apply mandatory eligibility filters."""

    group = [
        contractor
        for contractor in contractors
        if _normalize(contractor["city"]) == _normalize(query["city"])
        and _normalize(query["category"])
        in {_normalize(category) for category in contractor["categories"]}
    ]

    eligible: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    rejected_first_reason = {reason: 0 for reason in REJECTION_REASONS}
    for contractor in group:
        reasons = evaluate_candidate(contractor, query)
        if not reasons:
            eligible.append(deepcopy(contractor))
            continue
        primary_reason = reasons[0]
        rejected_first_reason[primary_reason] += 1
        rejections.append(
            {
                "id": contractor["id"],
                "anon_name": contractor["anon_name"],
                "primary_reason": primary_reason,
                "reasons": reasons,
            }
        )

    if len(group) != len(eligible) + sum(rejected_first_reason.values()):
        raise AssertionError("Eligibility accounting invariant was broken")
    return eligible, rejections, rejected_first_reason, len(group)


def _dataset_hash(contractors: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted(contractors, key=lambda item: str(item["id"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _base_response(
    query: dict[str, Any],
    contractors: list[dict[str, Any]],
    group_total: int,
    eligible_count: int,
    rejected_first_reason: dict[str, int],
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "success",
        "query": deepcopy(query),
        "results": [],
        "stats": {
            "city_category_total": group_total,
            "eligible_count": eligible_count,
            "returned_count": 0,
            "rejected_first_reason": deepcopy(rejected_first_reason),
        },
        "rejections": deepcopy(rejections),
        "message": "",
        "meta": {
            "ranking_method": "not_run",
            "dataset_hash": _dataset_hash(contractors),
            "scoring_version": "not_run",
        },
    }


def _no_match_message(counts: dict[str, int]) -> str:
    labels = {
        "busy": "заняты на выбранную дату",
        "over_budget": "имеют начальную цену выше бюджета",
        "wrong_format": "не берут выбранный формат мероприятия",
        "wrong_language": "не работают на выбранном языке",
        "too_short": "не подходят по длительности",
    }
    details = [
        f"{counts[reason]} {labels[reason]}"
        for reason in REJECTION_REASONS
        if counts[reason]
    ]
    return (
        "Кандидаты в этой категории есть, но ни один не прошёл обязательные условия. "
        "По первичной причине в порядке проверок: "
        + "; ".join(details)
        + "."
    )


def _success_message(eligible_count: int, group_total: int, returned_count: int) -> str:
    if eligible_count > 3:
        return f"Условия проходят {eligible_count}, показываем 3."
    if eligible_count < 3 and eligible_count == group_total:
        return (
            f"В этом городе в выбранной категории всего {eligible_count}; "
            "показываем все подходящие варианты."
        )
    if eligible_count < 3:
        return (
            f"Условия проходят {eligible_count}; показываем все. Остальные профили "
            "исключены обязательными фильтрами."
        )
    return f"Условия проходят {eligible_count}, показываем {returned_count}."


def _resolve_integrations(
    ranker: Ranker | None, card_builder: CardBuilder | None
) -> tuple[Ranker, CardBuilder, str, str]:
    if (ranker is None) != (card_builder is None):
        raise IntegrationContractError(
            "ranker and card_builder must be supplied together for an injected integration"
        )
    if ranker is None and card_builder is None:
        try:
            ranking_module = importlib.import_module("ranking")
        except ModuleNotFoundError as error:
            if error.name != "ranking":
                raise
            raise IntegrationNotReadyError(
                "Participant 2 module ranking.py is not available; expected "
                "rank_candidates(candidates, query) and build_cards(top_candidates, query)"
            ) from error
        try:
            ranker = ranking_module.rank_candidates
            card_builder = ranking_module.build_cards
        except AttributeError as error:
            raise IntegrationNotReadyError(
                "ranking.py must expose rank_candidates and build_cards"
            ) from error
        ranking_method = getattr(
            ranking_module, "RANKING_METHOD", f"{ranking_module.__name__}.rank_candidates"
        )
        scoring_version = getattr(ranking_module, "SCORING_VERSION", "unspecified")
        return ranker, card_builder, str(ranking_method), str(scoring_version)

    assert ranker is not None and card_builder is not None
    ranking_method = getattr(
        ranker, "ranking_method", f"{ranker.__module__}.{ranker.__name__}"
    )
    scoring_version = getattr(ranker, "scoring_version", "test-or-custom-adapter")
    return ranker, card_builder, str(ranking_method), str(scoring_version)


def _rank_and_build_cards(
    eligible: list[dict[str, Any]],
    query: dict[str, Any],
    ranker: Ranker,
    card_builder: CardBuilder,
) -> list[dict[str, Any]]:
    ranked = ranker(deepcopy(eligible), deepcopy(query))
    if not isinstance(ranked, list):
        raise IntegrationContractError("rank_candidates must return a list")
    eligible_ids = [candidate["id"] for candidate in eligible]
    try:
        ranked_ids = [candidate["id"] for candidate in ranked]
    except (KeyError, TypeError) as error:
        raise IntegrationContractError(
            "rank_candidates must return dictionaries containing id"
        ) from error
    if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != set(eligible_ids):
        raise IntegrationContractError(
            "rank_candidates must return every eligible candidate exactly once"
        )

    top_candidates = deepcopy(ranked[:3])
    cards = card_builder(top_candidates, deepcopy(query))
    if not isinstance(cards, list) or len(cards) != len(top_candidates):
        raise IntegrationContractError(
            "build_cards must return one card for every finalist"
        )
    for position, (card, finalist) in enumerate(zip(cards, top_candidates, strict=True)):
        if not isinstance(card, dict) or card.get("id") != finalist["id"]:
            raise IntegrationContractError(
                f"build_cards changed finalist order at position {position}"
            )
        if card.get("category") != query["category"]:
            raise IntegrationContractError(
                f"build_cards must set selected category at position {position}"
            )
        explanation = card.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise IntegrationContractError(
                f"build_cards must set a non-empty explanation at position {position}"
            )
        for field in (
            "anon_name",
            "city",
            "price_from_kzt",
            "synthetic",
            "city_imputed",
            "price_imputed",
        ):
            if card.get(field) != finalist.get(field):
                raise IntegrationContractError(
                    f"build_cards changed source field {field!r} at position {position}"
                )
    return deepcopy(cards)


def recommend(
    query: dict[str, Any],
    contractors: list[dict[str, Any]],
    *,
    ranker: Ranker | None = None,
    card_builder: CardBuilder | None = None,
) -> dict[str, Any]:
    """Validate, filter, rank and return the shared response contract.

    The keyword adapters exist for isolated core tests. Normal production calls use
    the public two-argument form and resolve Participant 2's ``ranking`` module.
    """

    validated_query = validate_query(query)
    validated_contractors = _validated_contractors(contractors)
    eligible, rejections, reason_counts, group_total = filter_candidates(
        validated_query, validated_contractors
    )
    response = _base_response(
        validated_query,
        validated_contractors,
        group_total,
        len(eligible),
        reason_counts,
        rejections,
    )

    if group_total == 0:
        response["status"] = "category_not_found"
        response["message"] = (
            f"В городе {validated_query['city']} нет подрядчиков категории "
            f"«{validated_query['category']}»."
        )
        return response

    if not eligible:
        response["status"] = "no_matching_candidates"
        response["message"] = _no_match_message(reason_counts)
        return response

    resolved_ranker, resolved_builder, ranking_method, scoring_version = (
        _resolve_integrations(ranker, card_builder)
    )
    cards = _rank_and_build_cards(
        eligible,
        validated_query,
        resolved_ranker,
        resolved_builder,
    )
    response["results"] = cards
    response["stats"]["returned_count"] = len(cards)
    response["message"] = _success_message(len(eligible), group_total, len(cards))
    response["meta"]["ranking_method"] = ranking_method
    response["meta"]["scoring_version"] = scoring_version
    return response

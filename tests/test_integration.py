from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from data_loader import load_contractors
from recommender import recommend


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "docs" / "hackathon dataset anonymized .csv"


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


SCENARIOS = [
    ("dense", {}, "success", 6, ["HK-88430", "HK-44923", "HK-29829"]),
    ("date_b", {"date": "2026-10-08"}, "success", 3,
     ["HK-29829", "HK-27222", "HK-44733"]),
    ("rare", {"date": "2026-10-09", "category": "Флорист", "event_type": "свадьба",
              "budget": 300_000}, "success", 2, ["HK-39372", "HK-90001"]),
    ("no_match", {"date": "2026-10-09", "budget": 100_000},
     "no_matching_candidates", 0, []),
    ("missing_category", {"date": "2026-10-09", "city": "Астана",
                          "category": "Инструменталист"}, "category_not_found", 0, []),
]


def scenario_query(overrides: dict) -> dict:
    return {
        **real_query(budget=1_000_000, event_type="корпоратив"),
        "date": "2026-10-06", **overrides,
    }


def assert_scenario(response, status, eligible, ordered_ids):
    assert response["status"] == status
    assert response["stats"]["eligible_count"] == eligible
    assert response["stats"]["returned_count"] == len(ordered_ids)
    assert [card["id"] for card in response["results"]] == ordered_ids
    assert response["stats"]["city_category_total"] == eligible + sum(
        response["stats"]["rejected_first_reason"].values()
    )
    assert response["message"]
    if ordered_ids:
        assert response["meta"]["scoring_version"] == "baseline-price-v1"
        assert all(card["explanation"] and card["evidence"] for card in response["results"])


@pytest.mark.parametrize(
    "name,overrides,status,eligible,ordered_ids", SCENARIOS, ids=[s[0] for s in SCENARIOS]
)
def test_real_demo_pipeline_is_repeatable(name, overrides, status, eligible, ordered_ids):
    contractors = load_contractors(DATASET)
    request = scenario_query(overrides)
    response = recommend(request, contractors)
    assert_scenario(response, status, eligible, ordered_ids)
    assert recommend(request, contractors) == response
    # Rejections follow input order; ranking and data fingerprint must not.
    reordered = recommend(request, list(reversed(contractors)))
    assert reordered["results"] == response["results"]
    assert reordered["meta"] == response["meta"]


def test_real_date_pair_explains_disappearing_finalists_by_calendar():
    contractors = load_contractors(DATASET)
    first = recommend(scenario_query({}), contractors)
    second = recommend(scenario_query({"date": "2026-10-08"}), contractors)
    disappeared = {card["id"] for card in first["results"]} - {
        card["id"] for card in second["results"]
    }
    assert disappeared == {"HK-88430", "HK-44923"}
    by_id = {profile["id"]: profile for profile in contractors}
    rejections = {entry["id"]: entry for entry in second["rejections"]}
    for identifier in disappeared:
        assert "2026-10-08" in by_id[identifier]["busy_dates"]
        assert rejections[identifier]["primary_reason"] == "busy"


@pytest.mark.parametrize("field", ["language", "event_type", "category", "city"])
def test_real_baseline_case_and_space_variants_keep_grounded_explanations(field):
    contractors = load_contractors(DATASET)
    request = scenario_query({})
    canonical = recommend(request, contractors)
    request[field] = "  " + request[field].upper() + "  "
    changed = recommend(request, contractors)
    assert changed["status"] == canonical["status"]
    assert changed["stats"] == canonical["stats"]
    assert changed["meta"] == canonical["meta"]
    assert [card["id"] for card in changed["results"]] == [card["id"] for card in canonical["results"]]
    for card in changed["results"]:
        assert f"язык «{changed['query']['language']}» указан" in card["explanation"]
        assert f"формат «{changed['query']['event_type']}»" in card["explanation"]


@pytest.mark.parametrize(
    "name,overrides,status,eligible,ordered_ids", SCENARIOS, ids=[s[0] for s in SCENARIOS]
)
def test_real_ui_uses_default_csv_and_production_pipeline(
    monkeypatch, name, overrides, status, eligible, ordered_ids
):
    monkeypatch.delenv("CONTRACTORS_DATA", raising=False)
    # Deliberately use another cwd: the default CSV must be relative to app.py.
    monkeypatch.chdir(ROOT / "tests")
    request = scenario_query(overrides)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    for key in ("city", "event_type", "category", "language"):
        app.selectbox(key=key).select(request[key])
    app.date_input(key="date").set_value(date.fromisoformat(request["date"]))
    app.number_input(key="budget").set_value(request["budget"])
    app.number_input(key="duration").set_value(float(request["duration"]))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    _, response, _ = app.session_state["last_result"]
    assert_scenario(response, status, eligible, ordered_ids)


def test_real_ui_respects_dataset_override_and_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACTORS_DATA", str(tmp_path / "absent.csv"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "Не удалось загрузить каталог" in app.error[0].value
    assert "last_result" not in app.session_state

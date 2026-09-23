"""UI contract tests. All profiles/responses here are synthetic test fixtures.

These tests do not verify real filtering, ranking or recommendation quality.
"""

from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest

from app import END_DATE, START_DATE, REASONS, catalog_options, make_query, money, validate_response


APP = Path(__file__).resolve().parents[1] / "app.py"
QUERY = dict(city="Алматы", date="2026-09-23", event_type="свадьба",
             category="Флорист", budget=300000, duration=None, language=None)


def profile(identifier, city="Алматы", category="Флорист"):
    return dict(
        id=identifier, anon_name=f"ТЕСТОВЫЙ профиль {identifier}",
        city=city, categories=[category], category=category, price_from_kzt=200000,
        event_formats=["свадьба"], languages=["русский"], max_hours=None,
        busy_dates=[], description="Синтетическая фикстура для UI-теста.",
        synthetic=True, city_imputed=True, price_imputed=True,
        explanation=f"ТЕСТ {identifier}: начальная цена 200 000 ₸; формат — свадьба.",
    )


def response(status="success", count=2):
    cards = [profile(identifier) for identifier in ["Z", "A", "M"][:count]] if status == "success" else []
    total = len(cards) if status == "success" else (0 if status == "category_not_found" else 2)
    rejected = dict.fromkeys(REASONS, 0)
    rejected["busy"] = total - len(cards)
    return dict(
        status=status, query=QUERY.copy(), results=cards,
        message={
            "success": f"ТЕСТ: подходят {len(cards)}; в городе больше профилей нет.",
            "category_not_found": "ТЕСТ: в городе нет этой категории.",
            "no_matching_candidates": "ТЕСТ: два профиля есть, оба заняты на дату.",
        }[status],
        stats=dict(city_category_total=total, eligible_count=len(cards),
                   returned_count=len(cards), rejected_first_reason=rejected),
        rejections=([dict(id="BUSY", anon_name="ТЕСТ занятый", primary_reason="busy", reasons=["busy"])]
                    if status == "no_matching_candidates" else []),
        meta=dict(ranking_method="synthetic-test-fixture", dataset_hash="test-only", scoring_version="test"),
    )


def start_app(monkeypatch, recommend=None, loader=None):
    data_module = ModuleType("data_loader")
    data_module.load_contractors = loader or (lambda path: [profile("Z"), profile("OTHER", "Астана", "Отель")])
    core_module = ModuleType("recommender")
    core_module.recommend = recommend or (lambda query, contractors: response())
    monkeypatch.setitem(sys.modules, "data_loader", data_module)
    monkeypatch.setitem(sys.modules, "recommender", core_module)
    return AppTest.from_file(str(APP)).run()


def submit(at):
    return at.button[0].click().run()


def test_form_uses_global_options_and_sends_none_and_iso(monkeypatch):
    calls = []
    def recommend(query, contractors):
        calls.append(deepcopy(query))
        return response()
    at = start_app(monkeypatch, recommend)
    assert not at.exception
    assert at.selectbox(key="category").options == ["Отель", "Флорист"]
    at.selectbox(key="category").select("Флорист")
    submit(at)
    assert calls == [QUERY]
    assert not at.exception


def test_selected_optional_fields_are_passed(monkeypatch):
    calls = []
    def recommend(query, contractors):
        calls.append(query)
        return response()
    at = start_app(monkeypatch, recommend)
    at.selectbox(key="language").select("русский")
    at.number_input(key="duration").set_value(6.0)
    at.date_input(key="date").set_value(date(2026, 12, 31))
    submit(at)
    assert calls[0]["language"] == "русский"
    assert calls[0]["duration"] == 6
    assert calls[0]["date"] == "2026-12-31"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_preserves_card_order_count_explanations_and_flags(monkeypatch, count):
    payload = response(count=count)
    at = start_app(monkeypatch, lambda q, c: deepcopy(payload))
    submit(at)
    assert not at.exception
    assert [s.value for s in at.subheader][1:] == [
        f"{i}. {card['anon_name']}" for i, card in enumerate(payload["results"], 1)
    ]
    captions = [element.value for element in at.caption]
    assert captions.count("Синтетический профиль") == count
    assert captions.count("Цена проставлена при подготовке данных") == count
    assert captions.count("Город проставлен при подготовке данных") == count
    for card in payload["results"]:
        assert card["explanation"] in [element.value for element in at.markdown]
    assert at.success[0].value == payload["message"]


@pytest.mark.parametrize("status", ["category_not_found", "no_matching_candidates"])
def test_empty_outcomes_are_distinct_and_keep_core_message(monkeypatch, status):
    payload = response(status)
    at = start_app(monkeypatch, lambda q, c: payload)
    submit(at)
    assert not at.exception
    assert not at.success and not at.error
    assert payload["message"] in [m.value for m in at.markdown]
    assert len(at.subheader) == 1  # query heading, no cards
    if status == "category_not_found":
        assert at.info[0].value == "В городе нет этой категории"
    else:
        assert at.warning[0].value == "Никто не прошёл условия запроса"
        assert "ТЕСТ занятый (BUSY): Заняты на дату" in [t.value for t in at.text]


def test_service_failure_clears_previous_result(monkeypatch):
    calls = []
    def recommend(query, contractors):
        calls.append(query)
        if len(calls) > 1:
            raise RuntimeError("Test failure, not a business result")
        return response()
    at = start_app(monkeypatch, recommend)
    submit(at)
    assert at.success
    submit(at)
    assert not at.success and not at.subheader
    assert "ошибка сервиса" in at.error[0].value
    assert not at.exception


def test_load_failure_is_not_empty_result(monkeypatch):
    def loader(path):
        raise FileNotFoundError("test-only")
    at = start_app(monkeypatch, loader=loader)
    assert "ошибка данных" in at.error[0].value
    assert not at.button and not at.warning and not at.exception


def test_missing_backend_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "recommender", None)
    at = AppTest.from_file(str(APP)).run()
    assert "ещё не подключён" in at.error[0].value
    assert not at.exception


@pytest.mark.parametrize("event_date", [START_DATE, END_DATE])
def test_inclusive_calendar_boundaries(event_date):
    query = make_query("Алматы", event_date, "свадьба", "Флорист", 1, None, None)
    assert query["date"] == event_date.isoformat()


@pytest.mark.parametrize("field,value", [
    ("event_date", date(2026, 9, 22)), ("event_date", date(2027, 1, 1)),
    ("budget", 0), ("budget", -1), ("budget", float("nan")),
    ("duration", 0), ("duration", -1), ("duration", float("inf")),
])
def test_invalid_inputs(field, value):
    args = dict(city="Алматы", event_date=START_DATE, event_type="свадьба",
                category="Флорист", budget=1, duration=None, language=None)
    args[field] = value
    with pytest.raises(ValueError):
        make_query(**args)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="unknown"),
    lambda r: r.update(results=[]),
    lambda r: r.update(message=""),
    lambda r: r["results"][0].update(synthetic="False"),
    lambda r: r["results"][0].update(explanation=""),
])
def test_broken_backend_contract_is_error(monkeypatch, mutate):
    payload = response()
    mutate(payload)
    at = start_app(monkeypatch, lambda q, c: payload)
    submit(at)
    assert "ошибка сервиса" in at.error[0].value
    assert not at.success and not at.exception


def test_catalog_options_do_not_mutate_profiles():
    catalog = [profile("1"), profile("2", "Астана", "Отель")]
    before = deepcopy(catalog)
    assert catalog_options(catalog)["category"] == ["Отель", "Флорист"]
    assert catalog == before


def test_price_is_not_rounded_and_integer_ids_are_supported():
    assert money(200000.50) == "200 000.5 ₸"
    assert money(200000) == "200 000 ₸"
    payload = response()
    payload["results"][0]["id"] = 7
    validate_response(payload)

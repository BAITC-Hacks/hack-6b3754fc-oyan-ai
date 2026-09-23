"""Participant 3: presentation only; selection belongs to recommender.py."""

from datetime import date
import logging
import math
import os
from pathlib import Path
from time import perf_counter

import streamlit as st


START_DATE = date(2026, 9, 23)
END_DATE = date(2026, 12, 31)
ROOT = Path(__file__).resolve().parent
REASONS = {
    "busy": "Заняты на дату",
    "over_budget": "Начальная цена выше бюджета",
    "wrong_format": "Не работают с этим форматом",
    "wrong_language": "Не поддерживают выбранный язык",
    "too_short": "Не хватает длительности присутствия",
}
LOGGER = logging.getLogger(__name__)


def catalog_options(contractors):
    """Use the entire catalog so a category absent in one city stays selectable."""
    return {
        "city": sorted({c["city"] for c in contractors}),
        **{
            key: sorted({value for c in contractors for value in c[field]})
            for key, field in (
                ("category", "categories"),
                ("event_type", "event_formats"),
                ("language", "languages"),
            )
        },
    }


def make_query(city, event_date, event_type, category, budget, duration, language):
    if not isinstance(event_date, date) or not START_DATE <= event_date <= END_DATE:
        raise ValueError("Выберите дату с 23 сентября по 31 декабря 2026 года.")
    if budget is None or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Бюджет должен быть больше нуля.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Длительность должна быть больше нуля или не указана.")
    return {
        "city": city,
        "date": event_date.isoformat(),
        "event_type": event_type,
        "category": category,
        "budget": budget,
        "duration": duration,
        "language": language,
    }


def money(value):
    from decimal import Decimal

    text = format(Decimal(str(value)), ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(",", " ") + " ₸"


def validate_response(response):
    """Reject broken response contracts, without repeating eligibility/ranking."""
    status = response["status"]
    cards = response["results"]
    if status not in {"success", "category_not_found", "no_matching_candidates"}:
        raise ValueError("Неизвестный статус ответа.")
    if not isinstance(cards, list) or not isinstance(response["message"], str):
        raise ValueError("Некорректная структура ответа.")
    if not response["message"].strip():
        raise ValueError("В ответе отсутствует объяснение результата.")
    if (status == "success" and not 1 <= len(cards) <= 3) or (
        status != "success" and cards
    ):
        raise ValueError("Количество карточек не соответствует статусу.")
    for card in cards:
        identifier = card["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError("В карточке отсутствует идентификатор.")
        for field in ("anon_name", "category", "city", "explanation"):
            if not isinstance(card[field], str) or not card[field].strip():
                raise ValueError("В карточке отсутствует обязательный текст.")
        if not math.isfinite(card["price_from_kzt"]) or card["price_from_kzt"] < 0:
            raise ValueError("В карточке некорректная цена.")
        for flag in ("synthetic", "city_imputed", "price_imputed"):
            if not isinstance(card[flag], bool):
                raise ValueError("В карточке некорректный флаг происхождения данных.")
    stats = response["stats"]
    for field in ("city_category_total", "eligible_count", "returned_count"):
        if not isinstance(stats[field], int) or stats[field] < 0:
            raise ValueError("Некорректная статистика ответа.")
    for reason in REASONS:
        count = stats["rejected_first_reason"][reason]
        if not isinstance(count, int) or count < 0:
            raise ValueError("Некорректная статистика исключений.")
    for rejection in response["rejections"]:
        if not all(field in rejection for field in ("id", "anon_name", "primary_reason", "reasons")):
            raise ValueError("Некорректная запись причины исключения.")
    if not isinstance(response["meta"], dict):
        raise ValueError("Некорректные сведения о подборе.")


def render_response(response):
    status = response["status"]
    if status == "success":
        st.success(response["message"])
    elif status == "category_not_found":
        st.info("В городе нет этой категории")
        st.write(response["message"])
    else:
        st.warning("Никто не прошёл условия запроса")
        st.write(response["message"])

    stats = response["stats"]
    cols = st.columns(3)
    cols[0].metric("В городе и категории", stats["city_category_total"])
    cols[1].metric("Проходят условия", stats["eligible_count"])
    cols[2].metric("Показано", stats["returned_count"])

    # Keep the exact order and wording supplied by the core and explanations.
    for index, card in enumerate(response["results"], start=1):
        with st.container(border=True):
            st.subheader(f"{index}. {card['anon_name']}")
            st.text(f"{card['category']} · {card['city']} · от {money(card['price_from_kzt'])}")
            if card["synthetic"]:
                st.caption("Синтетический профиль")
            else:
                st.caption("Профиль исходного каталога")
            if card["city_imputed"]:
                st.caption("Город проставлен при подготовке данных")
            if card["price_imputed"]:
                st.caption("Цена проставлена при подготовке данных")
            st.write(card["explanation"])
            st.caption(f"ID: {card['id']}")

    with st.expander("Почему остальные не показаны"):
        st.caption(
            "Числа переданы сервисом подбора. Каждый исключённый профиль учтён "
            "по первой причине: занятость → цена → формат → язык → длительность."
        )
        for code, label in REASONS.items():
            st.text(f"{label}: {stats['rejected_first_reason'][code]}")
        for rejection in response["rejections"]:
            reason = REASONS.get(rejection["primary_reason"], rejection["primary_reason"])
            st.text(f"{rejection['anon_name']} ({rejection['id']}): {reason}")

    with st.expander("Сведения о подборе"):
        st.json(response["meta"])


def render_form(contractors, recommend):
    options = catalog_options(contractors)
    if not contractors or any(not options[k] for k in ("city", "category", "event_type")):
        st.error("Каталог не содержит данных для формы. Это ошибка загрузки, а не результат подбора.")
        return
    with st.form("request"):
        left, right = st.columns(2)
        city = left.selectbox("Город", options["city"], key="city")
        event_date = right.date_input(
            "Дата мероприятия", value=START_DATE, min_value=START_DATE,
            max_value=END_DATE, format="DD.MM.YYYY", key="date",
        )
        event_type = left.selectbox("Тип мероприятия", options["event_type"], key="event_type")
        category = right.selectbox("Категория подрядчика", options["category"], key="category")
        budget = left.number_input("Бюджет, ₸", min_value=1, value=300000, step=10000, key="budget")
        language = right.selectbox(
            "Язык — необязательно", [None, *options["language"]],
            format_func=lambda value: "Не указан" if value is None else value,
            key="language",
        )
        duration = left.number_input(
            "Длительность, ч — необязательно", min_value=0.5, value=None,
            step=0.5, key="duration", placeholder="Не указана",
        )
        submitted = st.form_submit_button("Подобрать подрядчиков", type="primary")

    if submitted:
        # An error must never leave a previous successful result on screen.
        st.session_state.pop("last_result", None)
        try:
            query = make_query(city, event_date, event_type, category, budget, duration, language)
        except ValueError as exc:
            st.error(str(exc))
            return
        try:
            with st.spinner("Проверяем условия и готовим объяснения…"):
                started = perf_counter()
                response = recommend(query, contractors)
                elapsed = perf_counter() - started
                validate_response(response)
            st.session_state["last_result"] = (query, response, elapsed)
        except Exception:
            LOGGER.exception("Recommendation failed")
            st.error("Не удалось выполнить подбор. Это ошибка сервиса, а не отсутствие подходящих подрядчиков.")
            return

    if "last_result" in st.session_state:
        query, response, elapsed = st.session_state["last_result"]
        st.subheader("Результат отправленного запроса")
        st.text(
            f"{query['city']} · {query['date']} · {query['event_type']} · "
            f"{query['category']} · бюджет {money(query['budget'])}"
        )
        st.caption(
            f"Язык: {query['language'] or 'не указан'}; "
            f"длительность: {str(query['duration']) + ' ч' if query['duration'] is not None else 'не указана'}. "
            "После изменения условий нажмите кнопку подбора."
        )
        render_response(response)
        st.caption(f"Время этого вызова подбора: {elapsed:.3f} с (без запуска приложения и загрузки каталога).")


def main():
    st.set_page_config(page_title="Умный подбор подрядчиков", page_icon="🔎", layout="centered")
    st.title("Умный подбор подрядчиков")
    st.write("До трёх вариантов из каталога — с объяснением, почему они подходят вашему мероприятию.")
    st.caption(
        "Цена «от» — начальная стоимость, не итоговая смета. "
        "Доступность проверяется только по календарю каталога; это не бронирование."
    )
    try:
        from data_loader import load_contractors
        from recommender import recommend
    except ImportError:
        st.error("Сервис подбора ещё не подключён. Рекомендации пока недоступны.")
        return
    dataset_path = Path(os.environ.get("CONTRACTORS_DATA", ROOT / "hackathon-dataset-anonymized.jsonl"))
    try:
        contractors = load_contractors(dataset_path)
        catalog_options(contractors)
    except Exception:
        LOGGER.exception("Catalog loading failed")
        st.error("Не удалось загрузить каталог. Это ошибка данных, а не отсутствие подходящих подрядчиков.")
        return
    render_form(contractors, recommend)


if __name__ == "__main__":
    main()

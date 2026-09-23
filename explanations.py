"""Grounded template cards for the final candidates chosen by the core."""

from copy import deepcopy

from scoring import numeric_value
from semantic import select_description_excerpt


def collect_evidence(candidate, query):
    evidence = []
    for source, record, fields in (
        ("profile", candidate, ("price_from_kzt", "event_formats", "city", "max_hours",
                                "languages", "synthetic", "price_imputed", "city_imputed")),
        ("query", query, ("budget", "event_type", "category", "date", "duration", "language")),
    ):
        for field in fields:
            if field in record:
                evidence.append({"source": source, "field": field, "value": deepcopy(record[field])})
    excerpt = select_description_excerpt(candidate.get("description", ""), query)
    if excerpt:
        evidence.append({"source": "profile", "field": "description", **excerpt})
    return evidence


def _money(value):
    text = format(value, ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(",", " ")


def _explain(candidate, query):
    price = numeric_value(candidate["price_from_kzt"], "price_from_kzt")
    budget = numeric_value(query["budget"], "budget")
    difference = budget - price
    first = f"Начальная цена — от {_money(price)} ₸"
    if difference > 0:
        first += f", на {_money(difference)} ₸ ниже лимита {_money(budget)} ₸"
    elif difference == 0:
        first += f", равна лимиту {_money(budget)} ₸"
    else:
        # Defensive wording; eligibility still belongs exclusively to the core.
        first += f", выше лимита {_money(budget)} ₸ на {_money(-difference)} ₸"
    if candidate.get("price_imputed"):
        first += " (цена проставлена при подготовке данных)"
    event = query.get("event_type")
    if event in candidate.get("event_formats", []):
        first += f"; в каталоге указан формат «{event}»"
    if candidate.get("city_imputed"):
        first += f"; город «{candidate['city']}» проставлен при подготовке данных"
    first += "."

    excerpt = select_description_excerpt(candidate.get("description", ""), query)
    if excerpt:
        second = f"В описании указано: «{excerpt['value']}»."
    else:
        second = "Краткая цитата из описания не приведена; это не означает отсутствия опыта."
    return first + " " + second


def build_cards(top_candidates, query):
    """Preserve all finalists, their order, nested data and provenance flags.

    The caller slices rank_candidates(... )[:3] before calling this function.
    No filtering, sorting, truncation or external generation happens here.
    """
    cards = []
    for candidate in top_candidates:
        card = deepcopy(candidate)
        card["category"] = query["category"]
        card["explanation"] = _explain(candidate, query)
        card["evidence"] = collect_evidence(candidate, query)
        cards.append(card)
    # Equal texts are a data limitation, not a reason to invent distinctions.
    counts = {}
    for card in cards:
        counts[card["explanation"]] = counts.get(card["explanation"], 0) + 1
    for card in cards:
        if counts[card["explanation"]] > 1:
            card["explanation"] = card["explanation"].rstrip(".") + "; приведённые факты совпадают с другой показанной карточкой."
    return cards

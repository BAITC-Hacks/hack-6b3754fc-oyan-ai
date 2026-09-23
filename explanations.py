"""Grounded template cards for the final candidates chosen by the core."""

from copy import deepcopy

from scoring import numeric_value
from semantic import query_text, select_description_excerpt


def _semantic_evidence(candidate, query):
    facts = [fact for fact in candidate.get("evidence", []) if fact.get("role") == "semantic_scoring"]
    if not facts:
        if "semantic_similarity" in candidate.get("score_breakdown", {}):
            raise ValueError("semantic score is missing its evidence")
        return None
    if len(facts) != 1:
        raise ValueError("ambiguous semantic evidence")
    fact = facts[0]
    if (fact.get("source") != "profile" or fact.get("field") != "description"
            or fact.get("value") != candidate.get("description", "")
            or fact.get("query_text") != query_text(query)
            or fact.get("similarity") != candidate.get("score")
            or fact.get("similarity") != candidate.get("score_breakdown", {}).get("semantic_similarity")):
        raise ValueError("semantic evidence does not match the profile, query or score")
    return fact


def _description_evidence(candidate, query, semantic_fact):
    if not semantic_fact:
        excerpt = select_description_excerpt(candidate.get("description", ""), query)
        return {"source": "profile", "field": "description", **excerpt} if excerpt else None
    facts = [fact for fact in candidate.get("evidence", []) if fact.get("role") == "semantic_excerpt"]
    if len(facts) > 1:
        raise ValueError("ambiguous semantic excerpt")
    if not facts:
        return None
    fact = facts[0]
    start, end = fact.get("start"), fact.get("end")
    description = candidate.get("description", "")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(description)
            or fact.get("source") != "profile" or fact.get("field") != "description"
            or description[start:end] != fact.get("value")):
        raise ValueError("semantic excerpt is not a literal source fragment")
    return fact


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
    semantic_fact = _semantic_evidence(candidate, query)
    excerpt = _description_evidence(candidate, query, semantic_fact)
    if excerpt:
        evidence.append(deepcopy(excerpt))
    if semantic_fact:
        evidence.append(deepcopy(semantic_fact))
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
    semantic_fact = _semantic_evidence(candidate, query)
    if semantic_fact:
        first = f"Описание сравнено по смысловой близости к запросу «{query['category']}, {query['event_type']}»; начальная цена — от {_money(price)} ₸"
    else:
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

    excerpt = _description_evidence(candidate, query, semantic_fact)
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

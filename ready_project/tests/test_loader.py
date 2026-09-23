from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts import DatasetLoadError
from data_loader import load_contractors


DATASET = Path("docs/hackathon dataset anonymized .csv")


def valid_profile(identifier: str = "P-1") -> dict:
    return {
        "id": identifier,
        "anon_name": "Тестовый профиль",
        "categories": ["Ведущий"],
        "city": "Алматы",
        "city_imputed": False,
        "synthetic": True,
        "price_from_kzt": 100_000,
        "price_imputed": False,
        "event_formats": ["свадьба"],
        "languages": ["русский"],
        "max_hours": 6,
        "busy_dates": ["2026-10-01"],
        "description": "Профиль для изолированного теста загрузчика.",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_loads_real_csv() -> None:
    contractors = load_contractors(DATASET)

    assert len(contractors) == 66
    assert len({contractor["id"] for contractor in contractors}) == 66
    assert isinstance(contractors[0]["busy_dates"], list)
    assert isinstance(contractors[0]["synthetic"], bool)


def test_loads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    write_jsonl(path, [valid_profile()])

    contractors = load_contractors(path)

    assert contractors[0]["id"] == "P-1"
    assert contractors[0]["busy_dates"] == ["2026-10-01"]


def test_missing_file_names_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"

    with pytest.raises(DatasetLoadError, match=r"Dataset file not found:.*missing.jsonl"):
        load_contractors(path)


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    path.write_text(
        json.dumps(valid_profile(), ensure_ascii=False) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match=r"Invalid JSON.*line 2"):
        load_contractors(path)


def test_invalid_field_reports_line_and_field(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    profile = valid_profile()
    profile["price_from_kzt"] = 0
    write_jsonl(path, [profile])

    with pytest.raises(DatasetLoadError, match=r"line 1: field price_from_kzt"):
        load_contractors(path)


def test_duplicate_id_reports_both_lines(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    write_jsonl(path, [valid_profile(), valid_profile()])

    with pytest.raises(DatasetLoadError, match=r"Duplicate id 'P-1'.*line 2.*line 1"):
        load_contractors(path)


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "profiles.txt"
    path.write_text("data", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="expected .csv or .jsonl"):
        load_contractors(path)

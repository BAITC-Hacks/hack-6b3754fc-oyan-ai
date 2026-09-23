"""Load and validate the organizer-provided contractor dataset."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from contracts import Contractor, DatasetLoadError


LIST_FIELDS = ("categories", "event_formats", "languages", "busy_dates")
BOOLEAN_FIELDS = ("city_imputed", "synthetic", "price_imputed")
REQUIRED_FIELDS = tuple(Contractor.model_fields)


def _validation_message(error: ValidationError) -> str:
    problem = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in problem["loc"])
    return f"field {location}: {problem['msg']}"


def _parse_boolean(value: str, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"field {field}: expected true or false")


def _split_pipe(value: str, *, allow_empty: bool) -> list[str]:
    if not value.strip():
        return [] if allow_empty else [""]
    return [item.strip() for item in value.split("|")]


def _convert_csv_row(row: dict[str, str | None]) -> dict[str, Any]:
    converted: dict[str, Any] = dict(row)
    for field in LIST_FIELDS:
        raw_value = row.get(field)
        if raw_value is None:
            raise ValueError(f"field {field}: missing column value")
        converted[field] = _split_pipe(raw_value, allow_empty=field == "busy_dates")

    for field in BOOLEAN_FIELDS:
        raw_value = row.get(field)
        if raw_value is None:
            raise ValueError(f"field {field}: missing column value")
        converted[field] = _parse_boolean(raw_value, field)

    raw_price = row.get("price_from_kzt")
    try:
        converted["price_from_kzt"] = int(raw_price or "")
    except ValueError as error:
        raise ValueError("field price_from_kzt: expected integer") from error

    raw_hours = row.get("max_hours")
    if raw_hours is None or not raw_hours.strip():
        converted["max_hours"] = None
    else:
        try:
            converted["max_hours"] = float(raw_hours) if "." in raw_hours else int(raw_hours)
        except ValueError as error:
            raise ValueError("field max_hours: expected number or empty value") from error
    return converted


def _read_csv(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise DatasetLoadError(f"Dataset {path} has no CSV header")
            duplicate_headers = {
                name for name in reader.fieldnames if reader.fieldnames.count(name) > 1
            }
            if duplicate_headers:
                raise DatasetLoadError(
                    f"Dataset {path} has duplicate CSV columns: {sorted(duplicate_headers)}"
                )
            missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames))
            unexpected = sorted(set(reader.fieldnames) - set(REQUIRED_FIELDS))
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f"missing columns {missing}")
                if unexpected:
                    details.append(f"unexpected columns {unexpected}")
                raise DatasetLoadError(f"Invalid CSV header in {path}: {'; '.join(details)}")

            for row in reader:
                line_number = reader.line_num
                if None in row:
                    raise DatasetLoadError(
                        f"Invalid dataset {path} at line {line_number}: too many CSV values"
                    )
                try:
                    yield line_number, _convert_csv_row(row)
                except ValueError as error:
                    raise DatasetLoadError(
                        f"Invalid dataset {path} at line {line_number}: {error}"
                    ) from error
    except UnicodeDecodeError as error:
        raise DatasetLoadError(f"Dataset {path} is not valid UTF-8: {error}") from error
    except OSError as error:
        raise DatasetLoadError(f"Could not read dataset {path}: {error}") from error


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetLoadError(
                        f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(item, dict):
                    raise DatasetLoadError(
                        f"Invalid dataset {path} at line {line_number}: expected JSON object"
                    )
                yield line_number, item
    except UnicodeDecodeError as error:
        raise DatasetLoadError(f"Dataset {path} is not valid UTF-8: {error}") from error
    except OSError as error:
        raise DatasetLoadError(f"Could not read dataset {path}: {error}") from error


def _validated_records(
    path: Path, records: Iterable[tuple[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    contractors: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for line_number, raw_record in records:
        try:
            contractor = Contractor.model_validate(raw_record)
        except ValidationError as error:
            raise DatasetLoadError(
                f"Invalid dataset {path} at line {line_number}: {_validation_message(error)}"
            ) from error
        if contractor.id in seen_ids:
            raise DatasetLoadError(
                f"Duplicate id {contractor.id!r} in {path} at line {line_number}; "
                f"first seen at line {seen_ids[contractor.id]}"
            )
        seen_ids[contractor.id] = line_number
        contractors.append(contractor.model_dump(mode="json"))

    if not contractors:
        raise DatasetLoadError(f"Dataset {path} contains no contractor records")
    return contractors


def load_contractors(path: str | Path) -> list[dict[str, Any]]:
    """Load a UTF-8 CSV or JSONL dataset and validate every record."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetLoadError(f"Dataset file not found: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetLoadError(f"Dataset path is not a file: {dataset_path}")

    suffix = dataset_path.suffix.casefold()
    if suffix == ".csv":
        records = _read_csv(dataset_path)
    elif suffix == ".jsonl":
        records = _read_jsonl(dataset_path)
    else:
        raise DatasetLoadError(
            f"Unsupported dataset format {dataset_path.suffix!r} for {dataset_path}; "
            "expected .csv or .jsonl"
        )
    return _validated_records(dataset_path, records)

"""Shared data contracts for the contractor recommendation core."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
)


CALENDAR_START = date(2026, 9, 23)
CALENDAR_END = date(2026, 12, 31)

ALLOWED_CITIES = frozenset({"Алматы", "Астана", "Зарубежье"})
ALLOWED_EVENT_TYPES = frozenset(
    {"свадьба", "той", "корпоратив", "конференция", "юбилей", "день рождения"}
)
ALLOWED_LANGUAGES = frozenset({"русский", "казахский", "английский"})

REJECTION_REASONS = (
    "busy",
    "over_budget",
    "wrong_format",
    "wrong_language",
    "too_short",
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
PositiveNumber = Annotated[int | float, Field(gt=0)]


class DatasetLoadError(ValueError):
    """The dataset could not be read or violates the shared contract."""


class QueryValidationError(ValueError):
    """A recommendation query violates the shared contract."""


class IntegrationNotReadyError(RuntimeError):
    """Participant 2's ranking/card integration is not available yet."""


class IntegrationContractError(RuntimeError):
    """An integrated ranking or card builder broke the shared contract."""


def _casefold(value: str) -> str:
    return value.strip().casefold()


def _parse_iso_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("date must use ISO YYYY-MM-DD")
    stripped = value.strip()
    if len(stripped) != 10:
        raise ValueError("date must use ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(stripped)
    except ValueError as error:
        raise ValueError("date must use ISO YYYY-MM-DD") from error
    if parsed.isoformat() != stripped:
        raise ValueError("date must use ISO YYYY-MM-DD")
    return parsed


class Contractor(BaseModel):
    """Validated source profile. Original display spelling is preserved."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: NonEmptyStr
    anon_name: NonEmptyStr
    categories: Annotated[list[NonEmptyStr], Field(min_length=1)]
    city: NonEmptyStr
    city_imputed: StrictBool
    synthetic: StrictBool
    price_from_kzt: PositiveInt
    price_imputed: StrictBool
    event_formats: Annotated[list[NonEmptyStr], Field(min_length=1)]
    languages: Annotated[list[NonEmptyStr], Field(min_length=1)]
    max_hours: PositiveNumber | None
    busy_dates: list[date]
    description: NonEmptyStr

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        if _casefold(value) not in {_casefold(item) for item in ALLOWED_CITIES}:
            raise ValueError(f"unsupported city: {value}")
        return value

    @field_validator("event_formats")
    @classmethod
    def validate_event_formats(cls, values: list[str]) -> list[str]:
        allowed = {_casefold(item) for item in ALLOWED_EVENT_TYPES}
        invalid = [value for value in values if _casefold(value) not in allowed]
        if invalid:
            raise ValueError(f"unsupported event format(s): {', '.join(invalid)}")
        return values

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, values: list[str]) -> list[str]:
        allowed = {_casefold(item) for item in ALLOWED_LANGUAGES}
        invalid = [value for value in values if _casefold(value) not in allowed]
        if invalid:
            raise ValueError(f"unsupported language(s): {', '.join(invalid)}")
        return values

    @field_validator("busy_dates")
    @classmethod
    def validate_busy_dates(cls, values: list[date]) -> list[date]:
        outside = [value for value in values if not CALENDAR_START <= value <= CALENDAR_END]
        if outside:
            raise ValueError(
                "busy date outside supported calendar: "
                + ", ".join(value.isoformat() for value in outside)
            )
        return values

    @field_validator("busy_dates", mode="before")
    @classmethod
    def parse_busy_dates(cls, values: Any) -> list[date]:
        if not isinstance(values, list):
            raise ValueError("busy_dates must be a list")
        return [_parse_iso_date(value) for value in values]


class RecommendationQuery(BaseModel):
    """Validated public query contract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    city: NonEmptyStr
    date: date
    event_type: NonEmptyStr
    category: NonEmptyStr
    budget: PositiveInt
    duration: PositiveNumber | None = None
    language: NonEmptyStr | None = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_iso_date(cls, value: Any) -> date:
        return _parse_iso_date(value)

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        if _casefold(value) not in {_casefold(item) for item in ALLOWED_CITIES}:
            raise ValueError(f"unsupported city: {value}")
        return value

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        if _casefold(value) not in {_casefold(item) for item in ALLOWED_EVENT_TYPES}:
            raise ValueError(f"unsupported event type: {value}")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is not None and _casefold(value) not in {
            _casefold(item) for item in ALLOWED_LANGUAGES
        }:
            raise ValueError(f"unsupported language: {value}")
        return value

    @field_validator("date")
    @classmethod
    def validate_calendar_window(cls, value: date) -> date:
        if not CALENDAR_START <= value <= CALENDAR_END:
            raise ValueError(
                f"date must be between {CALENDAR_START.isoformat()} "
                f"and {CALENDAR_END.isoformat()}"
            )
        return value

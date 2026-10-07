# ABOUTME: Coerces model-returned amounts, dates, currency codes and optional strings into canonical forms.
# ABOUTME: Handles US and European number formats, ISO/day-first/textual EN and ES dates; raises ValueError on anything else.
import math
import re
from datetime import date
from typing import Any, Optional

_MONTHS = {
    # English
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # Spanish
    "enero": 1, "ene": 1, "febrero": 2, "marzo": 3, "abril": 4, "abr": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "ago": 8, "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12, "dic": 12,
}
_CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP"}
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "-", "—"}

_ISO_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$")
# Currency signs, known currency codes and spaces may surround an amount; anything else is not an amount.
_CURRENCY_DECORATION = re.compile(
    r"\b(?:USD|EUR|GBP|MXN|CAD|AUD|CHF|JPY|COP|CLP|ARS)\b|[$€£\s]", re.IGNORECASE
)
_NUMBER_SHAPE = re.compile(r"[-(]?[\d.,]*\d[\d.,]*\)?")
_THOUSANDS_GROUPS = re.compile(r"^\d{1,3}([.,]\d{3})+$")


def optional_text(value: Any) -> Optional[str]:
    """Strip a string; map empty and null-like strings to None. Numbers become their string form."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected text, got boolean {value!r}")
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {type(value).__name__}")
    value = value.strip()
    return None if value.lower() in _NULL_STRINGS else value


def parse_amount(value: Any) -> float:
    """Parse a number or a printed amount ('$1,032.54', '1.437,12', '634,73 €') into a float rounded to 2 places.

    A string with both separators takes the right-most one as the decimal point. A lone separator followed by
    exactly three digits, or repeated separators, is a thousands separator (invoice amounts carry 0 or 2
    decimals); any other lone separator is the decimal point.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(f"expected an amount, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        stripped = _CURRENCY_DECORATION.sub("", text)
        if not _NUMBER_SHAPE.fullmatch(stripped):
            raise ValueError(f"not a number: {value!r}")
        negative = stripped.startswith("-") or stripped.startswith("(")
        core = re.sub(r"[^\d.,]", "", stripped)
        if "." in core and "," in core:
            decimal = "." if core.rfind(".") > core.rfind(",") else ","
            thousands = "," if decimal == "." else "."
            core = core.replace(thousands, "").replace(decimal, ".")
        elif "." in core or "," in core:
            if _THOUSANDS_GROUPS.match(core) and not core.startswith("0"):
                core = re.sub(r"[.,]", "", core)
            elif core.count(".") + core.count(",") == 1:
                core = core.replace(",", ".")
            else:
                raise ValueError(f"ambiguous number format: {value!r}")
        try:
            number = float(core)
        except ValueError as exc:
            raise ValueError(f"not a number: {value!r}") from exc
        if negative:
            number = -number
    else:
        raise ValueError(f"expected an amount, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"amount is not finite: {value!r}")
    return round(number, 2)


def parse_quantity(value: Any) -> float:
    """Parse a quantity; whole numbers come back as floats with no fractional part (3 -> 3.0)."""
    number = parse_amount(value)
    if number < 0:
        raise ValueError(f"quantity is negative: {value!r}")
    return number


def _make_date(year: int, month: int, day: int, original: Any) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValueError(f"not a calendar date: {original!r}") from exc


def parse_date(value: Any) -> Optional[str]:
    """Return an ISO date string (YYYY-MM-DD) or None for an empty value.

    Accepts ISO; numeric dates, read day-first ('12/07/2026' is 12 July) unless only month-first is a valid
    reading ('07/25/2026'); and textual dates in English or Spanish ('January 21, 2026', '2 May 2026',
    '17 de mayo de 2026').
    """
    text = optional_text(value)
    if text is None:
        return None
    iso = _ISO_DATE.match(text)
    if iso:
        return _make_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), value)
    numeric = _NUMERIC_DATE.match(text)
    if numeric:
        first, second, year = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
        if second > 12 >= first:
            return _make_date(year, first, second, value)
        return _make_date(year, second, first, value)
    tokens = [t for t in re.sub(r"[,.]", " ", text.lower()).split() if t not in {"de", "del"}]
    if len(tokens) == 3:
        years = [t for t in tokens if re.fullmatch(r"\d{4}", t)]
        months = [t for t in tokens if t in _MONTHS]
        days = [t for t in tokens if re.fullmatch(r"\d{1,2}", t)]
        if len(years) == 1 and len(months) == 1 and len(days) == 1:
            return _make_date(int(years[0]), _MONTHS[months[0]], int(days[0]), value)
    raise ValueError(f"unrecognised date: {value!r}")


def parse_currency(value: Any) -> Optional[str]:
    """Return an upper-case three-letter currency code, mapping the euro and pound signs; None for empty."""
    text = optional_text(value)
    if text is None:
        return None
    if text in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[text]
    if re.fullmatch(r"[A-Za-z]{3}", text):
        return text.upper()
    raise ValueError(f"unrecognised currency (expected a 3-letter code): {value!r}")

# ABOUTME: Unit tests for the model-response coercion helpers: amounts, dates, currency codes and optional text.
# ABOUTME: Covers US and European number formats, ES and EN date forms, and values that must be rejected.
import pytest

from ap_invoice_processor.llm.coercion import (
    optional_text,
    parse_amount,
    parse_currency,
    parse_date,
    parse_quantity,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (12, 12.0),
        (12.345, 12.35),
        ("985.84", 985.84),
        ("$1,032.54", 1032.54),
        ("£1,221.09", 1221.09),
        ("1.437,12", 1437.12),
        ("1.437,12 €", 1437.12),
        ("634,73 €", 634.73),
        ("7,35", 7.35),
        ("1,437", 1437.0),
        ("1.437", 1437.0),
        ("1.437.112,50", 1437112.5),
        ("1,437,112.50", 1437112.5),
        ("USD 20,359.43", 20359.43),
        ("$17,551.23 MXN", 17551.23),
        ("0,50", 0.5),
        ("0.125", 0.12),
        ("(12.50)", -12.5),
        ("-3.00", -3.0),
    ],
)
def test_parse_amount_accepts(raw, expected):
    assert parse_amount(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    ["", "abc", "17 de mayo de 2026", "mar 5 2026", "1.2.3,4,5", None, True, [1], float("nan"), float("inf"), "1e5x"],
)
def test_parse_amount_rejects(raw):
    with pytest.raises(ValueError):
        parse_amount(raw)


def test_parse_quantity_rejects_negative():
    assert parse_quantity("3") == 3.0
    with pytest.raises(ValueError):
        parse_quantity(-1)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-01-21", "2026-01-21"),
        ("2026/1/5", "2026-01-05"),
        ("12/07/2026", "2026-07-12"),  # day-first
        ("07/25/2026", "2026-07-25"),  # only month-first is a valid reading
        ("27.07.2026", "2026-07-27"),
        ("January 21, 2026", "2026-01-21"),
        ("21 Jan 2026", "2026-01-21"),
        ("2 May 2026", "2026-05-02"),
        ("Sept 3, 2026", "2026-09-03"),
        ("17 de mayo de 2026", "2026-05-17"),
        ("1 de julio de 2026", "2026-07-01"),
        ("7 de junio de 2026", "2026-06-07"),
        ("  2026-02-13  ", "2026-02-13"),
        (None, None),
        ("", None),
        ("n/a", None),
    ],
)
def test_parse_date_accepts(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["31/02/2026", "2026-13-01", "yesterday", "May 2026", "32 de mayo de 2026", "12", 20260121, "2026-1-1-1"])
def test_parse_date_rejects(raw):
    with pytest.raises(ValueError):
        parse_date(raw)


@pytest.mark.parametrize("raw,expected", [("usd", "USD"), (" EUR ", "EUR"), ("€", "EUR"), ("£", "GBP"), (None, None), ("", None)])
def test_parse_currency_accepts(raw, expected):
    assert parse_currency(raw) == expected


@pytest.mark.parametrize("raw", ["$", "dollars", "US", "12", 5])
def test_parse_currency_rejects(raw):
    with pytest.raises(ValueError):
        parse_currency(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), ("  ", None), ("null", None), ("None", None), ("N/A", None), ("-", None), (" PO-7048 ", "PO-7048"), (123, "123")],
)
def test_optional_text(raw, expected):
    assert optional_text(raw) == expected


@pytest.mark.parametrize("raw", [True, ["x"], {"a": 1}])
def test_optional_text_rejects(raw):
    with pytest.raises(ValueError):
        optional_text(raw)

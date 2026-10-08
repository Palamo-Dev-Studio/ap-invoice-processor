# ABOUTME: Tests for fixture-backed invoice extraction: all 40 corpus documents, validation errors, and prompt isolation.
# ABOUTME: Isolation tests prove the prompt carries reader text only and that ground-truth inputs are rejected.
import glob
import json
import os
import re
import shutil

import pytest

from ap_invoice_processor.llm.coercion import parse_amount, parse_date
from ap_invoice_processor.llm.extraction import (
    ExtractedInvoice,
    ExtractionError,
    ForbiddenInputError,
    build_extraction_prompt,
    extract_invoice,
)
from ap_invoice_processor.llm.provider import FixtureProvider, LLMProviderError
from ap_invoice_processor.reader import ReaderOutput, read_document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
READER_TEXT_DIR = os.path.join(FIXTURES, "reader_text")
EXTRACT_DIR = os.path.join(FIXTURES, "llm", "extract")

DOC_IDS = sorted(
    [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CORPUS, "pdf", "*.pdf"))]
    + [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CORPUS, "images", "*.png"))]
)


def _reader_output(doc_id):
    with open(os.path.join(READER_TEXT_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
        text = f.read()
    method = "tesseract" if re.search(r"-(scan|photo)$", doc_id) else "pdftotext"
    return ReaderOutput(doc_id=doc_id, text=text, method=method, ocr_lang="eng" if method == "tesseract" else None)


def _fixture(doc_id):
    with open(os.path.join(EXTRACT_DIR, f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def _gt(doc_id):
    with open(os.path.join(CORPUS, "ground_truth", f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)


PROVIDER = FixtureProvider(os.path.join(ROOT, "tests", "fixtures", "llm"))


def test_every_corpus_document_has_reader_text_and_an_extraction_fixture():
    assert len(DOC_IDS) == 40
    for doc_id in DOC_IDS:
        assert os.path.isfile(os.path.join(READER_TEXT_DIR, f"{doc_id}.txt")), doc_id
        assert os.path.isfile(os.path.join(EXTRACT_DIR, f"{doc_id}.json")), doc_id
    assert len(glob.glob(os.path.join(EXTRACT_DIR, "*.json"))) == 40


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_extraction_fixture_validates_and_is_marked_as_authored_from_reader_text(doc_id):
    assert _fixture(doc_id)["_authored_from"] == "reader_text"
    inv = extract_invoice(_reader_output(doc_id), PROVIDER)
    assert isinstance(inv, ExtractedInvoice)
    assert inv.total > 0
    assert inv.line_items
    assert inv.invoice_number


@pytest.mark.skipif(not (shutil.which("pdftotext") and shutil.which("tesseract")), reason="reader binaries not installed")
@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_saved_reader_text_matches_the_live_reader(doc_id):
    # The saved text, and the fixtures authored from it, come from English-only OCR; the reader's default now also
    # loads Spanish, so this check pins the English-only reading explicitly.
    kind = "images" if re.search(r"-(scan|photo)$", doc_id) else "pdf"
    ext = "png" if kind == "images" else "pdf"
    live = read_document(os.path.join(CORPUS, kind, f"{doc_id}.{ext}"), ocr_langs="eng")
    assert live.text == _reader_output(doc_id).text


# --- audit: every fixture value must be traceable to its reader text (guards against fixtures copied from ground truth) ---

_NUMBER_TOKEN = re.compile(r"\d[\d.,]*\d|\d")
_DATE_TOKENS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
    re.compile(r"\d{1,2} de [^\W\d_]+ de \d{4}"),
    re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}"),
    re.compile(r"\d{1,2} [A-Z][a-z]{2} \d{4}"),
]


def _amounts_in(text):
    found = set()
    for token in _NUMBER_TOKEN.findall(text):
        try:
            found.add(parse_amount(token))
        except ValueError:
            pass
    return found


def _dates_in(text):
    found = set()
    for pattern in _DATE_TOKENS:
        for token in pattern.findall(text):
            try:
                found.add(parse_date(token))
            except ValueError:
                pass
    return found


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_fixture_values_are_traceable_to_reader_text(doc_id):
    text = _reader_output(doc_id).text
    fx = _fixture(doc_id)
    for key in ("vendor_name", "invoice_number", "po_number"):
        if fx[key] is not None:
            assert fx[key] in text, (doc_id, key, fx[key])
    amounts = _amounts_in(text)
    for key in ("subtotal", "tax", "total"):
        assert parse_amount(fx[key]) in amounts, (doc_id, key)
    dates = _dates_in(text)
    for key in ("invoice_date", "due_date"):
        if fx[key] is not None:
            assert parse_date(fx[key]) in dates, (doc_id, key, fx[key])
    for item in fx["line_items"]:
        assert item["description"] in text, (doc_id, item["description"])
        assert parse_amount(item["unit_price"]) in amounts
        assert parse_amount(item["amount"]) in amounts


def test_traceability_audit_would_catch_a_value_not_in_the_text():
    text = _reader_output("en-001").text
    assert 985.84 in _amounts_in(text)
    assert 985.85 not in _amounts_in(text)
    assert "2026-03-07" in _dates_in(text)
    assert "2026-03-08" not in _dates_in(text)


def test_ocr_damaged_fixtures_reflect_the_damage():
    damaged = {
        "en-017-photo": ("vendor_name", "Harborv"),
        "es-002-scan": ("vendor_name", "Servicios Tecnolégicos Alborada S.A."),
        "es-003-photo": ("vendor_name", "Papeleria El Roble S.A."),
        "es-007-scan": ("vendor_name", "Consultoria Estrella del Norte S.C."),
        "es-011-photo": ("vendor_name", "Imprenta Sol"),
        "en-004-scan": ("invoice_date", None),
    }
    for doc_id, (field, value) in damaged.items():
        assert _fixture(doc_id)[field] == value
        assert "_damage_note" in _fixture(doc_id)
        assert getattr(extract_invoice(_reader_output(doc_id), PROVIDER), field) == value


def test_european_and_textual_formats_in_fixtures_are_coerced():
    es4 = extract_invoice(_reader_output("es-004"), PROVIDER)
    assert es4.line_items[1].amount == pytest.approx(1437.12)
    assert es4.total == pytest.approx(1782.96)
    es1 = extract_invoice(_reader_output("es-001"), PROVIDER)
    assert (es1.invoice_date, es1.due_date) == ("2026-07-12", "2026-07-27")
    assert es1.total == pytest.approx(875.62)
    assert extract_invoice(_reader_output("es-002"), PROVIDER).invoice_date == "2026-05-17"
    assert extract_invoice(_reader_output("en-010"), PROVIDER).invoice_date == "2026-09-28"
    assert extract_invoice(_reader_output("en-012"), PROVIDER).total == pytest.approx(1391.42)


def test_null_po_stays_null():
    assert extract_invoice(_reader_output("en-002"), PROVIDER).po_number is None


# --- error handling ---


class StubProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, task, prompt, doc_id):
        self.calls.append((task, prompt, doc_id))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


GOOD = {"vendor_name": "V", "invoice_number": "N1", "total": "$10.00", "line_items": []}
RO = ReaderOutput(doc_id="doc-x", text="Some invoice text, total $10.00", method="pdftotext")


def test_valid_minimal_response():
    inv = extract_invoice(RO, StubProvider(GOOD))
    assert inv.total == 10.0 and inv.po_number is None and inv.line_items == []


def test_missing_fixture_becomes_a_structured_provider_error(tmp_path):
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(RO, FixtureProvider(str(tmp_path)))
    assert exc.value.stage == "provider" and exc.value.doc_id == "doc-x"
    assert "no fixture" in exc.value.to_dict()["message"]


def test_provider_error_is_wrapped():
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(RO, StubProvider(LLMProviderError("boom")))
    assert exc.value.stage == "provider"


def test_a_provider_that_is_not_enabled_is_not_swallowed():
    with pytest.raises(NotImplementedError):
        extract_invoice(RO, StubProvider(NotImplementedError("provider not enabled")))


@pytest.mark.parametrize("response", [[1], "text", None, 5])
def test_non_object_response(response):
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(RO, StubProvider(response))
    assert exc.value.stage == "response_shape"


@pytest.mark.parametrize(
    "mutation,field",
    [
        ({"total": None}, "total"),
        ({"total": "n/a"}, "total"),
        ({"invoice_date": "31/02/2026"}, "invoice_date"),
        ({"due_date": "someday"}, "due_date"),
        ({"currency": "$"}, "currency"),
        ({"subtotal": "abc"}, "subtotal"),
        ({"vendor_name": ["x"]}, "vendor_name"),
        ({"line_items": [{"description": "", "quantity": 1, "unit_price": 1, "amount": 1}]}, "line_items.0.description"),
        ({"line_items": [{"description": "x", "quantity": 1, "unit_price": "oops", "amount": 1}]}, "line_items.0.unit_price"),
        ({"line_items": [{"description": "x", "quantity": -2, "unit_price": 1, "amount": 1}]}, "line_items.0.quantity"),
        ({"line_items": [{"description": "x", "unit_price": 1, "amount": 1}]}, "line_items.0.quantity"),
    ],
)
def test_invalid_fields_produce_structured_validation_error(mutation, field):
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(RO, StubProvider({**GOOD, **mutation}))
    err = exc.value
    assert err.stage == "validation"
    assert field in [e["field"] for e in err.errors]
    assert err.to_dict()["errors"] == err.errors


def test_missing_total_key_is_a_validation_error():
    response = {k: v for k, v in GOOD.items() if k != "total"}
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(RO, StubProvider(response))
    assert exc.value.stage == "validation"


def test_blank_reader_text_is_an_input_error_and_provider_is_not_called():
    stub = StubProvider(GOOD)
    with pytest.raises(ExtractionError) as exc:
        extract_invoice(ReaderOutput(doc_id="d", text="  \n", method="pdftotext"), stub)
    assert exc.value.stage == "input" and stub.calls == []


def test_unknown_response_keys_are_ignored():
    inv = extract_invoice(RO, StubProvider({**GOOD, "ground_truth": {"x": 1}, "confidence": 0.9}))
    assert not hasattr(inv, "ground_truth")


# --- prompt isolation ---


def _scalars(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _scalars(v)
    elif isinstance(value, list):
        for v in value:
            yield from _scalars(v)
    elif value is not None:
        yield str(value)


def leaked_values(prompt, reader_text, ground_truth):
    """Ground-truth scalar values (4+ characters) that are absent from the reader text yet present in the prompt."""
    return sorted(
        {s for s in _scalars(ground_truth) if len(s) >= 4 and s not in reader_text and s in prompt}
    )


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_prompt_contains_reader_text_and_no_ground_truth_only_value(doc_id):
    ro = _reader_output(doc_id)
    stub = StubProvider(GOOD)
    extract_invoice(ro, stub)
    (task, prompt, sent_doc_id), = stub.calls
    assert task == "extract" and sent_doc_id == doc_id
    assert prompt == build_extraction_prompt(ro.text)
    assert ro.text in prompt
    gt = _gt(doc_id)
    assert leaked_values(prompt, ro.text, gt) == []
    # The fixed instructions alone must not hold any ground-truth value either.
    assert leaked_values(build_extraction_prompt(""), "", gt) == []


def test_isolation_check_has_teeth():
    ro = _reader_output("en-001")
    gt = _gt("en-001")
    leaky = build_extraction_prompt(ro.text) + json.dumps(gt)
    assert leaked_values(leaky, ro.text, gt), "the leak detector must flag a prompt carrying ground truth"
    assert "2026-03-07" in leaked_values(build_extraction_prompt(ro.text) + "due 2026-03-07", ro.text, gt)


@pytest.mark.parametrize("key", ["ground_truth", "simulated_extraction", "raw_text", "id", "anything_else"])
def test_mapping_with_extra_keys_is_rejected(key):
    payload = {"doc_id": "d", "text": "t", "method": "pdftotext", key: {"vendor_name": "X"}}
    stub = StubProvider(GOOD)
    with pytest.raises(ForbiddenInputError, match=key):
        extract_invoice(payload, stub)
    assert stub.calls == []


def test_mapping_with_only_reader_fields_is_accepted():
    stub = StubProvider(GOOD)
    extract_invoice({"doc_id": "d", "text": "Total $10.00", "method": "pdftotext"}, stub)
    assert len(stub.calls) == 1


def test_reader_output_carrying_extra_attributes_is_rejected():
    ro = ReaderOutput(doc_id="d", text="t", method="pdftotext")
    ro.__dict__["ground_truth"] = {"a": 1}
    with pytest.raises(ForbiddenInputError):
        extract_invoice(ro, StubProvider(GOOD))


def test_non_reader_input_is_rejected():
    with pytest.raises(ForbiddenInputError):
        extract_invoice("raw text", StubProvider(GOOD))


# USD and MXN both accept "$", so a USD<->MXN swap in a fixture is not caught by this trace.
_CURRENCY_MARKS = {"USD": ("USD", "$"), "GBP": ("GBP", "£"), "EUR": ("EUR", "€"), "MXN": ("MXN", "MX$", "$")}


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_fixture_currency_is_traceable_to_reader_text(doc_id):
    fx = _fixture(doc_id)
    assert any(mark in _reader_output(doc_id).text for mark in _CURRENCY_MARKS[fx["currency"]]), (doc_id, fx["currency"])

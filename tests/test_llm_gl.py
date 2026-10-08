# ABOUTME: Tests for the LLM GL coder: fixtures for all 40 corpus documents, validation, and keyword-coder fallback.
# ABOUTME: Covers an off-chart account, provider failure, malformed responses, and prompt contents.
import glob
import json
import os
from types import SimpleNamespace

import pytest

from ap_invoice_processor.keyword_coder import keyword_code, match_vendor
from ap_invoice_processor.llm.extraction import extract_invoice
from ap_invoice_processor.llm.gl import GLCode, build_gl_prompt, code_lines, load_chart
from ap_invoice_processor.llm.provider import FixtureProvider, LLMProviderError
from ap_invoice_processor.reader import ReaderOutput
from ap_invoice_processor.skill_loader import load_skill_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
GL_DIR = os.path.join(FIXTURES, "llm", "gl")
EXTRACT_DIR = os.path.join(FIXTURES, "llm", "extract")
PROVIDER = FixtureProvider(os.path.join(FIXTURES, "llm"))
CHART = load_chart()
CHART_NUMBERS = {a["account_number"] for a in CHART}
DOC_IDS = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(EXTRACT_DIR, "*.json")))


def _invoice(doc_id):
    with open(os.path.join(FIXTURES, "reader_text", f"{doc_id}.txt"), encoding="utf-8") as f:
        text = f.read()
    return extract_invoice(ReaderOutput(doc_id=doc_id, text=text, method="pdftotext"), PROVIDER)


class StubProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, task, prompt, doc_id):
        self.calls.append((task, prompt, doc_id))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


LINES = [{"description": "Social media ad spend"}, {"description": "Packing tape, case of 36"}]


def _ok(i, account="6500", confidence=0.9, reason="r"):
    return {"line": i, "account": account, "confidence": confidence, "reason": reason}


def test_chart_loads_five_accounts():
    assert CHART_NUMBERS == {"6000", "6100", "6200", "6500", "7000"}


def test_every_corpus_document_has_a_gl_fixture_matching_its_extraction_fixture():
    assert len(DOC_IDS) == 40
    total = 0
    for doc_id in DOC_IDS:
        with open(os.path.join(GL_DIR, f"{doc_id}.json"), encoding="utf-8") as f:
            gl = json.load(f)
        assert gl["_authored_from"] == "reader_text"
        assert [e["line"] for e in gl["lines"]] == list(range(len(_invoice(doc_id).line_items))), doc_id
        total += len(gl["lines"])
    assert len(glob.glob(os.path.join(GL_DIR, "*.json"))) == 40
    assert total == 98


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_fixture_coding_is_valid_and_uses_the_llm_path(doc_id):
    inv = _invoice(doc_id)
    codes = code_lines(inv.line_items, CHART, PROVIDER, doc_id, vendor_name=inv.vendor_name)
    assert len(codes) == len(inv.line_items)
    for code in codes:
        assert code.source == "llm"
        assert code.account in CHART_NUMBERS
        assert 0 <= code.confidence <= 1
        assert code.reason
        assert code.account_name == next(a["account_name"] for a in CHART if a["account_number"] == code.account)


def test_off_chart_account_falls_back_for_that_line_only():
    stub = StubProvider({"lines": [_ok(0), _ok(1, account="9999")]})
    codes = code_lines(LINES, CHART, stub, "d", vendor_name="Northgate Ridge Supply Co.")
    assert codes[0].source == "llm" and codes[0].account == "6500"
    assert codes[1].source == "keyword_fallback"
    assert codes[1].account == "6100"
    assert "'9999' is not in the chart" in codes[1].reason
    expected = keyword_code("Northgate Ridge Supply Co.", "Packing tape, case of 36", None, load_skill_rules())
    assert codes[1].account == expected.gl


def test_fallback_uses_the_vendor_master_and_chart_name():
    stub = StubProvider({"lines": [_ok(0, account="0000")]})
    (code,) = code_lines([{"description": "Cloud EC2"}], CHART, stub, "d", vendor_name="Amazon Web Services")
    assert (code.account, code.account_name, code.source) == ("6000", "Cloud & Hosting Services", "keyword_fallback")
    assert code.department == "Engineering" and code.confidence == 0.7
    assert "vendor_master" in code.reason


def test_default_rule_fallback_has_low_confidence():
    (code,) = code_lines([{"description": "zzz"}], CHART, StubProvider({"lines": [_ok(0, account="0")]}), "d", vendor_name="Nobody")
    assert code.account == "6100" and code.confidence == 0.3 and "default" in code.reason


def test_missing_fixture_sends_every_line_to_the_keyword_coder(tmp_path):
    codes = code_lines(LINES, CHART, FixtureProvider(str(tmp_path)), "d", vendor_name="Acme Marketing Solutions")
    assert [c.source for c in codes] == ["keyword_fallback"] * 2
    assert all("provider failed" in c.reason and "no fixture" in c.reason for c in codes)
    rules = load_skill_rules()
    with open(os.path.join(ROOT, "data", "vendor_master.json"), encoding="utf-8") as f:
        vm = match_vendor("Acme Marketing Solutions", json.load(f))
    assert [c.account for c in codes] == [keyword_code("Acme Marketing Solutions", l["description"], vm, rules).gl for l in LINES]


def test_provider_error_falls_back():
    codes = code_lines(LINES, CHART, StubProvider(LLMProviderError("down")), "d")
    assert all(c.source == "keyword_fallback" for c in codes)


def test_a_provider_that_is_not_enabled_is_not_swallowed():
    with pytest.raises(NotImplementedError):
        code_lines(LINES, CHART, StubProvider(NotImplementedError("provider not enabled")), "d")


@pytest.mark.parametrize("response", [[1], "x", None, {}, {"lines": "nope"}, {"lines": None}])
def test_unusable_response_shape_falls_back_for_every_line(response):
    codes = code_lines(LINES, CHART, StubProvider(response), "d")
    assert [c.source for c in codes] == ["keyword_fallback"] * 2


@pytest.mark.parametrize(
    "entry,why",
    [
        ({"line": 0, "account": 6500, "confidence": 0.9, "reason": "r"}, "not in the chart"),
        ({"line": 0, "account": None, "confidence": 0.9, "reason": "r"}, "not in the chart"),
        ({"line": 0, "account": "6500", "confidence": 1.5, "reason": "r"}, "confidence"),
        ({"line": 0, "account": "6500", "confidence": -0.1, "reason": "r"}, "confidence"),
        ({"line": 0, "account": "6500", "confidence": "high", "reason": "r"}, "confidence"),
        ({"line": 0, "account": "6500", "confidence": True, "reason": "r"}, "confidence"),
        ({"line": 0, "account": "6500", "confidence": 0.9, "reason": "  "}, "reason"),
        ({"line": 0, "account": "6500", "confidence": 0.9}, "reason"),
    ],
)
def test_malformed_entry_falls_back_for_that_line(entry, why):
    codes = code_lines(LINES, CHART, StubProvider({"lines": [entry, _ok(1)]}), "d")
    assert codes[0].source == "keyword_fallback" and why in codes[0].reason
    assert codes[1].source == "llm"


def test_missing_entry_and_non_object_entries_fall_back():
    codes = code_lines(LINES, CHART, StubProvider({"lines": [_ok(0), "junk", {"line": "1"}]}), "d")
    assert codes[0].source == "llm"
    assert codes[1].source == "keyword_fallback" and "no entry" in codes[1].reason


def test_reason_is_trimmed_and_confidence_boundaries_accepted():
    codes = code_lines(LINES, CHART, StubProvider({"lines": [_ok(0, confidence=0, reason=" " + "x" * 500), _ok(1, confidence=1)]}), "d")
    assert [c.confidence for c in codes] == [0.0, 1.0]
    assert len(codes[0].reason) == 300 and not codes[0].reason.startswith(" ")


def test_prompt_holds_the_chart_and_descriptions_only():
    stub = StubProvider({"lines": [_ok(0), _ok(1)]})
    code_lines(LINES, CHART, stub, "doc-7", vendor_name="Northgate Ridge Supply Co.")
    (task, prompt, doc_id), = stub.calls
    assert task == "gl" and doc_id == "doc-7"
    assert prompt == build_gl_prompt([l["description"] for l in LINES], CHART)
    assert "Northgate" not in prompt
    for a in CHART:
        assert a["account_number"] in prompt and a["account_name"] in prompt
    assert "0: Social media ad spend" in prompt and "1: Packing tape, case of 36" in prompt


def test_lines_may_be_strings_mappings_or_objects():
    stub = StubProvider({"lines": [_ok(0), _ok(1), _ok(2)]})
    codes = code_lines(["a", {"description": "b"}, SimpleNamespace(description="c")], CHART, stub, "d")
    assert len(codes) == 3
    assert "0: a" in stub.calls[0][1] and "1: b" in stub.calls[0][1] and "2: c" in stub.calls[0][1]


def test_no_lines_returns_empty_without_calling_the_provider():
    stub = StubProvider({"lines": []})
    assert code_lines([], CHART, stub, "d") == [] and stub.calls == []


def test_glcode_rejects_unknown_source():
    with pytest.raises(ValueError):
        GLCode(account="6000", account_name="x", confidence=0.5, reason="r", source="magic")

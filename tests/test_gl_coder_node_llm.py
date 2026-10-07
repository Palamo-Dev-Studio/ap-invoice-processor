# ABOUTME: Node-level tests for the GL-Coder node's document path: LLM codes with a per-line source in the audit trail.
# ABOUTME: Also pins that the simulated-extraction path and the keyword fallback behave as before.
import json
import os
import shutil

import pytest

from ap_invoice_processor import document_intake
from ap_invoice_processor.models import InvoiceState, LineItem
from ap_invoice_processor.nodes import extractor_node, gl_coder_node, intake_node
from ap_invoice_processor.reader import ReaderOutput

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
LLM_FIXTURES = os.path.join(FIXTURES, "llm")
PDF = os.path.join(CORPUS, "pdf", "en-001.pdf")

needs_pdftotext = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")


class DummySession:
    id = "test-session"


class DummyContext:
    def __init__(self):
        self.state = {}
        self.session = DummySession()
        self.node_path = "Workflow/test"
        self.node = None
        self.run_id = "run-1"
        self.attempt_count = 1
        self.resume_inputs = {}
        self.interrupt_ids = []
        self.output = None
        self.route = None


@pytest.fixture(autouse=True)
def _fixture_provider(monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", LLM_FIXTURES)


@pytest.fixture
def saved_reader(monkeypatch):
    """Serve the saved reader text instead of shelling out, so these tests need no binaries."""

    def read(path):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(FIXTURES, "reader_text", f"{doc_id}.txt"), encoding="utf-8") as f:
            return ReaderOutput(doc_id=doc_id, text=f.read(), method="pdftotext")

    monkeypatch.setattr(document_intake, "read_document", read)


def _run_nodes(payload):
    ctx = DummyContext()
    state = intake_node._func(ctx, payload).output
    state = extractor_node._func(ctx, state).output
    return gl_coder_node._func(ctx, state).output


def _gl_step(state):
    (step,) = [s for s in state["decision_trail"] if s["node_name"] == "GL-Coder"]
    return step


def _fixture_gl(doc_id):
    with open(os.path.join(LLM_FIXTURES, "gl", f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)["lines"]


def _assert_llm_trail(state, doc_id):
    expected = _fixture_gl(doc_id)
    items = state["extracted_fields"]["line_items"]
    step = _gl_step(state)
    lines = step["output_summary"]["lines"]
    assert len(items) == len(expected) == len(lines) == 3
    for item, want, got in zip(items, expected, lines):
        assert item["gl_account"] == want["account"]
        assert item["gl_account_name"]
        assert got["source"] == "llm"
        assert got["account"] == want["account"]
        assert got["confidence"] == want["confidence"]
        assert got["reason"] == want["reason"]
    assert step["action"] == "Code Line Items via LLM Provider with Keyword Fallback"
    assert step["output_summary"]["llm_coded"] == 3
    assert step["output_summary"]["keyword_fallback"] == 0
    assert step["output_summary"]["provider"] == "FixtureProvider"


def test_document_payload_is_coded_by_the_llm_coder_with_source_in_the_trail(saved_reader):
    state = _run_nodes({"id": "INV-DOC-1", "document_path": PDF})
    assert state["document_id"] == "en-001"
    _assert_llm_trail(state, "en-001")


@needs_pdftotext
def test_document_payload_through_the_real_reader_is_coded_by_the_llm_coder():
    state = _run_nodes({"id": "INV-DOC-2", "document_path": PDF})
    _assert_llm_trail(state, "en-001")


def test_off_chart_account_in_the_fixture_falls_back_to_the_keyword_coder_for_that_line(
    saved_reader, tmp_path, monkeypatch
):
    for task in ("extract", "gl"):
        os.makedirs(tmp_path / task)
        shutil.copy(os.path.join(LLM_FIXTURES, task, "en-001.json"), tmp_path / task / "en-001.json")
    with open(tmp_path / "gl" / "en-001.json", encoding="utf-8") as f:
        gl = json.load(f)
    gl["lines"][1]["account"] = "9999"
    with open(tmp_path / "gl" / "en-001.json", "w", encoding="utf-8") as f:
        json.dump(gl, f)
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", str(tmp_path))

    state = _run_nodes({"document_path": PDF})
    step = _gl_step(state)
    sources = [line["source"] for line in step["output_summary"]["lines"]]
    assert sources == ["llm", "keyword_fallback", "llm"]
    fallback = step["output_summary"]["lines"][1]
    assert "9999" in fallback["reason"] and fallback["reason"].startswith("keyword fallback")
    assert fallback["account"] != "9999"
    assert state["extracted_fields"]["line_items"][1]["gl_account"] == fallback["account"]
    assert (step["output_summary"]["llm_coded"], step["output_summary"]["keyword_fallback"]) == (2, 1)


def test_missing_gl_fixture_sends_every_line_to_the_keyword_fallback(saved_reader, tmp_path, monkeypatch):
    os.makedirs(tmp_path / "extract")
    shutil.copy(os.path.join(LLM_FIXTURES, "extract", "en-001.json"), tmp_path / "extract" / "en-001.json")
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", str(tmp_path))

    state = _run_nodes({"document_path": PDF})
    lines = _gl_step(state)["output_summary"]["lines"]
    assert [line["source"] for line in lines] == ["keyword_fallback"] * 3
    assert all("provider failed" in line["reason"] for line in lines)
    assert all(item["gl_account"] for item in state["extracted_fields"]["line_items"])


def test_live_provider_stub_surfaces_at_the_gl_coder_not_a_silent_fallback(monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    state = InvoiceState(invoice_id="x", document_id="en-001")
    state.extracted_fields.vendor_name = "Harborview Print & Signage"
    state.extracted_fields.line_items = [LineItem(description="Foam board posters", unit_price=1.0, amount=1.0)]
    with pytest.raises(NotImplementedError):
        gl_coder_node._func(DummyContext(), state.model_dump())


def test_unreadable_document_codes_zero_lines_without_calling_the_provider(tmp_path):
    state = _run_nodes({"id": "INV-DOC-3", "document_path": str(tmp_path / "missing.pdf")})
    step = _gl_step(state)
    assert step["output_summary"]["lines"] == []
    assert step["output_summary"]["coded_line_items"] == 0
    assert step["confidence"] == 0.0


def _simulated_payload():
    return {
        "id": "INV-SIM-1",
        "raw_text": "raw",
        "simulated_extraction": {
            "vendor_name": "Amazon Web Services",
            "total_amount": 12.5,
            "line_items": [{"description": "Cloud EC2", "unit_price": 12.5, "amount": 12.5}],
        },
    }


def test_simulated_extraction_path_keeps_the_keyword_coder_and_its_trail_shape():
    state = _run_nodes(_simulated_payload())
    assert state["document_id"] is None
    (item,) = state["extracted_fields"]["line_items"]
    assert item["gl_account"] == "6000"
    step = _gl_step(state)
    assert step["action"] == "Apply Portable Agent Skill GL Rules"
    assert set(step["output_summary"]) == {"coded_line_items", "matched_vendor"}
    assert "source" not in json.dumps(step)
    assert step["confidence"] == 0.95


def test_simulated_extraction_path_never_consults_the_provider(monkeypatch):
    # With the live stub selected, a provider call would raise; the simulated path must not make one.
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    state = _run_nodes(_simulated_payload())
    assert state["extracted_fields"]["line_items"][0]["gl_account"] == "6000"

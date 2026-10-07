# ABOUTME: Tests for the optional document-intake path: reader -> fixture extraction -> InvoiceState, node and graph level.
# ABOUTME: Also pins that payloads without document_path keep the original simulated-extraction behaviour.
import asyncio
import json
import os
import shutil

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from ap_invoice_processor.document_intake import fill_state_from_document
from ap_invoice_processor.graph import root_agent
from ap_invoice_processor.hitl import is_paused_at_gate
from ap_invoice_processor.llm.provider import FixtureProvider
from ap_invoice_processor.models import InvoiceState
from ap_invoice_processor.nodes import extractor_node, gl_coder_node, intake_node, policy_validator_node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
PDF = os.path.join(CORPUS, "pdf", "en-001.pdf")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "llm")

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
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", FIXTURES)


@needs_pdftotext
def test_document_payload_is_read_and_extracted():
    event = intake_node._func(DummyContext(), {"id": "INV-DOC-1", "document_path": PDF})
    out = event.output
    fields = out["extracted_fields"]
    assert out["invoice_id"] == "INV-DOC-1"
    assert "Harborview Print & Signage" in out["raw_text"]
    assert fields["vendor_name"] == "Harborview Print & Signage"
    assert fields["invoice_number"] == "HP90867"
    assert fields["date"] == "2026-01-21"
    assert fields["po_number"] == "PO-7048"
    assert fields["total_amount"] == 985.84
    assert [(li["description"], li["qty"]) for li in fields["line_items"]][:2] == [
        ("Foam board posters 24x36", 11),
        ("Business cards, box of 500", 5),
    ]
    assert out["field_confidence"]["vendor_name"] == 1.0
    (step,) = out["decision_trail"]
    assert step["action"] == "Read Document & Extract Fields via LLM Provider"
    assert step["output_summary"]["extraction"] == "ok"
    assert step["output_summary"]["provider"] == "FixtureProvider"
    assert step["output_summary"]["reader_method"] == "pdftotext"


@needs_pdftotext
def test_document_id_becomes_invoice_id_when_payload_has_none():
    out = intake_node._func(DummyContext(), {"document_path": PDF}).output
    assert out["invoice_id"] == "en-001"
    assert out["decision_trail"][0]["output_summary"]["invoice_id"] == "en-001"


@needs_pdftotext
def test_simulated_extraction_is_ignored_when_a_document_is_supplied():
    sim = {"vendor_name": "SIMULATED", "total_amount": 1.0}
    out = intake_node._func(DummyContext(), {"id": "X", "document_path": PDF, "simulated_extraction": sim}).output
    assert out["extracted_fields"]["vendor_name"] == "Harborview Print & Signage"
    assert out["extracted_fields"]["total_amount"] == 985.84


def test_payload_without_document_path_keeps_the_simulated_extraction_behaviour():
    payload = {
        "id": "INV-SIM-1",
        "raw_text": "raw",
        "simulated_extraction": {
            "vendor_name": "Amazon Web Services",
            "total_amount": 12.5,
            "line_items": [{"description": "Cloud", "unit_price": 12.5, "amount": 12.5}],
            "confidence": {"vendor_name": 0.5},
        },
    }
    out = intake_node._func(DummyContext(), payload).output
    assert out["raw_text"] == "raw"
    assert out["extracted_fields"]["vendor_name"] == "Amazon Web Services"
    assert out["extracted_fields"]["line_items"][0]["description"] == "Cloud"
    assert out["field_confidence"]["vendor_name"] == 0.5
    (step,) = out["decision_trail"]
    assert step["action"] == "Pull & Normalize Raw Invoice"


@needs_pdftotext
def test_missing_fixture_yields_empty_fields_zero_confidence_and_human_review(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", str(tmp_path))
    ctx = DummyContext()
    state = intake_node._func(ctx, {"id": "INV-DOC-2", "document_path": PDF}).output
    summary = state["decision_trail"][0]["output_summary"]
    assert summary["extraction"] == "error" and summary["error"]["stage"] == "provider"
    assert state["extracted_fields"]["vendor_name"] is None and state["extracted_fields"]["line_items"] == []
    assert state["field_confidence"] == {k: 0.0 for k in state["field_confidence"]}
    state = extractor_node._func(ctx, state).output
    state = gl_coder_node._func(ctx, state).output
    event = policy_validator_node._func(ctx, state)
    assert event.actions.route == "human_review"
    assert event.output["validation_flags"]["low_confidence_fields"] is True


def test_unreadable_document_is_recorded_not_raised(tmp_path):
    out = intake_node._func(DummyContext(), {"id": "INV-DOC-3", "document_path": str(tmp_path / "missing.pdf")}).output
    summary = out["decision_trail"][0]["output_summary"]
    assert summary["extraction"] == "error" and summary["error"].startswith("reader:")
    assert out["field_confidence"]["total_amount"] == 0.0


@needs_pdftotext
def test_live_provider_stub_surfaces_as_an_error_not_a_silent_fallback(monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(NotImplementedError):
        intake_node._func(DummyContext(), {"id": "INV-DOC-4", "document_path": PDF})


@needs_pdftotext
def test_fractional_quantity_is_carried_as_one_unit_at_the_line_amount():
    class FractionalProvider:
        def complete(self, task, prompt, doc_id):
            return {
                "vendor_name": "V", "invoice_number": "N", "total": 30,
                "line_items": [{"description": "Hours", "quantity": 1.5, "unit_price": 20, "amount": 30}],
            }

    state = InvoiceState(invoice_id="x")
    fill_state_from_document(state, PDF, FractionalProvider())
    (item,) = state.extracted_fields.line_items
    assert (item.qty, item.unit_price, item.amount) == (1, 30.0, 30.0)


@needs_pdftotext
def test_explicit_provider_argument_is_used():
    state = InvoiceState(invoice_id="x")
    summary = fill_state_from_document(state, PDF, FixtureProvider(FIXTURES))
    assert summary["extraction"] == "ok" and summary["line_items"] == 3


@needs_pdftotext
def test_full_graph_pauses_at_the_human_gate_for_a_document_payload():
    """Runs the real workflow on a PDF payload; the unknown vendor must route to the human gate."""

    async def run():
        runner = InMemoryRunner(app=App(name="doc_app", root_agent=root_agent))
        session = await runner.session_service.create_session(app_name="doc_app", user_id="u")
        msg = types.Content(role="user", parts=[types.Part.from_text(text=json.dumps({"document_path": PDF}))])
        paused, states = False, []
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
            if event.output and isinstance(event.output, dict) and "invoice_id" in event.output:
                states.append(event.output)
            if is_paused_at_gate(event):
                paused = True
                break
        return paused, states

    paused, states = asyncio.run(run())
    assert paused
    final = states[-1]
    assert final["invoice_id"] == "en-001"
    assert final["route_signal"] == "human_review"
    assert final["validation_flags"]["unknown_vendor"] is True
    assert [s["node_name"] for s in final["decision_trail"]] == ["Intake", "Extractor", "GL-Coder", "Policy-Validator"]


@needs_pdftotext
def test_fields_the_extraction_left_null_get_zero_confidence():
    class SparseProvider:
        def complete(self, task, prompt, doc_id):
            return {"vendor_name": None, "invoice_number": None, "invoice_date": None, "total": 5, "line_items": []}

    state = InvoiceState(invoice_id="x")
    fill_state_from_document(state, PDF, SparseProvider())
    conf = state.field_confidence
    assert (conf.vendor_name, conf.invoice_number, conf.date, conf.line_items) == (0.0, 0.0, 0.0, 0.0)
    assert conf.total_amount == 1.0
    assert state.extracted_fields.total_amount == 5.0

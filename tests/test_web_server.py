# ABOUTME: Tests for the dashboard's document routes (/api/run-upload, /api/run-sample), demo mode and the per-request guards.
# ABOUTME: Runs the real ADK workflow through FastAPI's TestClient with the fixture provider; no network and no live model.
import asyncio
import io
import json
import logging
import os
import re
import shutil
import threading
import time

import httpx2
import pytest
from fastapi.testclient import TestClient

from ap_invoice_processor import reader
from ap_invoice_processor.llm import anthropic_provider
from ap_invoice_processor.llm.provider import FixtureProvider
from document_builders import make_pdf, png_header
from web import server, uploads
from web.presentation import DEMO_NOTICE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "llm")
needs_pdftotext = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
needs_ocr = pytest.mark.skipif(not (shutil.which("tesseract") and shutil.which("pdftoppm")), reason="OCR binaries not installed")

PDF_BYTES = b"%PDF-1.4\n" + b"x" * 64
SENTINEL_KEY = "sk-ant-WEB-SENTINEL-not-a-real-key"


@pytest.fixture(autouse=True)
def fixture_provider_and_private_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("AP_LLM_FIXTURES_DIR", FIXTURES)
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(staging_root))
    monkeypatch.setattr(server, "_documents_in_flight", 0)
    return staging_root


@pytest.fixture
def staging(fixture_provider_and_private_tmp):
    return fixture_provider_and_private_tmp


@pytest.fixture
def client():
    # Entering the client keeps one event loop alive, so the workflow task a request starts can finish.
    with TestClient(server.web_app) as c:
        yield c


def _wait_for(client, session_id, statuses, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/sessions/{session_id}").json()
        if state["status"] in statuses:
            return state
        time.sleep(0.2)
    raise AssertionError(f"session {session_id} never reached {statuses}; last state: {state['status']}")


def _upload(client, name, data, content_type="application/octet-stream"):
    return client.post("/api/run-upload", files={"file": (name, io.BytesIO(data), content_type)})


def _corpus_bytes(sub, name):
    with open(os.path.join(CORPUS, sub, name), "rb") as f:
        return f.read()


# --- config and demo mode ----------------------------------------------------------------------------------------------


def test_config_reports_the_provider_limits_and_demo_flag(client, monkeypatch):
    body = client.get("/api/config").json()
    assert body == {
        "demo_mode": False,
        "notice": None,
        "llm_provider": "fixture",
        "max_upload_bytes": 5 * 1024 * 1024,
        "accepted_types": [".jpeg", ".jpg", ".pdf", ".png"],
    }
    monkeypatch.setenv("AP_DEMO_MODE", "1")
    assert client.get("/api/config").json()["notice"] == DEMO_NOTICE


def test_config_never_carries_a_secret(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    assert SENTINEL_KEY not in client.get("/api/config").text


def test_the_page_carries_the_roi_banner_outside_demo_mode(client):
    page = client.get("/").text
    assert "roi-metrics-card" in page and "$14.50" in page and "87.9%" in page
    assert DEMO_NOTICE not in page
    assert "<!--demo-notice-->" not in page


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", " yes ", "on"])
def test_demo_mode_removes_the_roi_banner_and_shows_the_fixed_notice(client, monkeypatch, flag):
    monkeypatch.setenv("AP_DEMO_MODE", flag)
    page = client.get("/").text
    assert DEMO_NOTICE == "Test build · synthetic data only · results are not financial advice"
    assert DEMO_NOTICE in page
    for figure in ("roi-metrics-card", "$14.50", "$1.75", "87.9%", "Manual Processing", "Cost Savings", "roi-banner"):
        assert figure not in page, figure


@pytest.mark.parametrize("flag", ["", "0", "false", "no", "off", "2"])
def test_other_values_leave_demo_mode_off(client, monkeypatch, flag):
    monkeypatch.setenv("AP_DEMO_MODE", flag)
    assert "$14.50" in client.get("/").text


def test_the_static_assets_carry_no_hard_coded_roi_figures():
    for name in ("app.js", "style.css"):
        text = open(os.path.join(ROOT, "web", "static", name), encoding="utf-8").read()
        assert "87.9" not in text and "14.50" not in text and "1.75" not in text, name


# --- samples ----------------------------------------------------------------------------------------------------------


def test_the_sample_list_is_served(client):
    samples = client.get("/api/samples").json()
    assert len(samples) == 40
    assert {"id": "en-001.pdf", "label": "en-001 (PDF)", "kind": "PDF"} in samples


@needs_pdftotext
def test_a_sample_runs_the_document_path_to_the_human_gate_then_posts_on_approval(client, staging):
    started = client.post("/api/run-sample", json={"sample_id": "en-001.pdf"})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "started" and body["invoice_id"] == "en-001"

    state = _wait_for(client, body["session_id"], {"paused", "completed", "error"})
    assert state["status"] == "paused", state
    invoice = state["invoice_state"]
    assert invoice["document_id"] == "en-001"
    assert invoice["extracted_fields"]["vendor_name"] and invoice["extracted_fields"]["total_amount"] > 0
    trail = {step["node_name"]: step for step in invoice["decision_trail"]}
    assert trail["Intake"]["action"] == "Read Document & Extract Fields via LLM Provider"
    assert trail["Intake"]["output_summary"]["reader_method"] == "pdftotext"
    lines = trail["GL-Coder"]["output_summary"]["lines"]
    assert lines and all(line["reason"] and line["source"] in ("llm", "keyword_fallback") for line in lines)
    assert trail["Policy-Validator"]["output_summary"]["route_signal"] == "human_review"
    assert state["node_subtitles"]["GL-Coder"] == "Fixture coder + keyword fallback"
    assert state["node_subtitles"]["Intake"] == "Read document + extract"

    # The staged copy is gone once the first leg ended, and the corpus original is untouched.
    assert os.listdir(staging) == []
    assert os.path.isfile(os.path.join(CORPUS, "pdf", "en-001.pdf"))

    resumed = client.post(f"/api/sessions/{body['session_id']}/triage", json={"decision": "approved", "reasoning": "checked"})
    assert resumed.status_code == 200
    final = _wait_for(client, body["session_id"], {"completed", "aborted", "error"})
    assert final["status"] == "completed"
    assert final["invoice_state"]["posted_entry_id"].startswith("NS-POST-")
    assert os.listdir(staging) == []


@needs_ocr
def test_a_scanned_sample_runs_through_ocr(client, staging):
    body = client.post("/api/run-sample", json={"sample_id": "en-004-scan.png"}).json()
    state = _wait_for(client, body["session_id"], {"paused", "completed", "error"})
    assert state["status"] in ("paused", "completed"), state
    intake = state["invoice_state"]["decision_trail"][0]["output_summary"]
    assert intake["reader_method"] == "tesseract"
    assert os.listdir(staging) == []


@pytest.mark.parametrize("sample_id", ["../README.md", "/etc/passwd", "pdf/en-001.pdf", "unknown.pdf", ""])
def test_an_unknown_sample_is_a_404_and_stages_nothing(client, staging, sample_id):
    response = client.post("/api/run-sample", json={"sample_id": sample_id})
    assert response.status_code == 404
    assert os.listdir(staging) == []


# --- uploads: validation ----------------------------------------------------------------------------------------------


def test_an_oversize_upload_is_a_413_and_stages_nothing(client, staging):
    response = _upload(client, "big.pdf", PDF_BYTES + b"0" * (5 * 1024 * 1024))
    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]
    assert os.listdir(staging) == []


def test_a_file_just_over_the_limit_is_refused_after_reading(client, staging):
    response = _upload(client, "big.pdf", PDF_BYTES + b"0" * (5 * 1024 * 1024 - len(PDF_BYTES) + 1))
    assert response.status_code == 413
    assert os.listdir(staging) == []


@pytest.mark.parametrize(
    "name,data",
    [("notes.txt", b"hello"), ("run.exe", b"MZ\x90"), ("fake.pdf", b"MZ\x90\x00 not a pdf"), ("page.html", b"<html>")],
)
def test_a_disallowed_type_is_a_415_and_stages_nothing(client, staging, name, data):
    response = _upload(client, name, data)
    assert response.status_code == 415
    assert os.listdir(staging) == []


def test_an_empty_upload_is_a_400(client, staging):
    assert _upload(client, "empty.pdf", b"").status_code == 400
    assert os.listdir(staging) == []


def test_a_request_without_a_file_is_a_400(client, staging):
    assert client.post("/api/run-upload").status_code == 400
    assert client.post("/api/run-upload", data={"note": "no file here"}).status_code == 400
    assert client.post("/api/run-upload", json={"file": "x"}).status_code == 400
    assert os.listdir(staging) == []


def test_the_file_field_must_be_named_file(client, staging):
    response = client.post("/api/run-upload", files={"document": ("a.pdf", io.BytesIO(PDF_BYTES), "application/pdf")})
    assert response.status_code == 400
    assert os.listdir(staging) == []


def test_one_request_carries_one_document(client, staging):
    """The guard that stops a request from fanning out: two files, or a file plus a field, never start a run."""
    two_files = client.post(
        "/api/run-upload",
        files=[
            ("file", ("a.pdf", io.BytesIO(PDF_BYTES), "application/pdf")),
            ("file", ("b.pdf", io.BytesIO(PDF_BYTES), "application/pdf")),
        ],
    )
    assert two_files.status_code == 400
    file_and_field = client.post(
        "/api/run-upload", files={"file": ("a.pdf", io.BytesIO(PDF_BYTES), "application/pdf")}, data={"extra": "1"}
    )
    assert file_and_field.status_code == 400
    assert os.listdir(staging) == []
    assert server._documents_in_flight == 0


# --- uploads: processing and cleanup ----------------------------------------------------------------------------------


@needs_pdftotext
def test_an_uploaded_corpus_invoice_runs_the_document_path_and_the_file_is_deleted(client, staging):
    response = _upload(client, "en-002.pdf", _corpus_bytes("pdf", "en-002.pdf"), "application/pdf")
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    state = _wait_for(client, session_id, {"paused", "completed", "error"})
    assert state["status"] in ("paused", "completed"), state
    invoice = state["invoice_state"]
    assert invoice["document_id"] == "en-002"
    assert [s["node_name"] for s in invoice["decision_trail"]][:4] == ["Intake", "Extractor", "GL-Coder", "Policy-Validator"]
    assert os.listdir(staging) == []
    assert server._documents_in_flight == 0


@needs_pdftotext
def test_an_upload_with_no_fixture_goes_to_the_human_gate_with_empty_fields(client, staging):
    """Offline mode has no model: a file that is not a sample is routed to a human, never filled in."""
    response = _upload(client, "mystery invoice.pdf", _corpus_bytes("pdf", "en-001.pdf"), "application/pdf")
    state = _wait_for(client, response.json()["session_id"], {"paused", "completed", "error"})
    assert state["status"] == "paused"
    invoice = state["invoice_state"]
    assert invoice["document_id"] == "mystery_invoice"
    assert invoice["extracted_fields"]["vendor_name"] is None
    assert invoice["validation_flags"]["low_confidence_fields"] is True
    assert os.listdir(staging) == []


@needs_pdftotext
def test_a_corrupt_pdf_with_valid_magic_is_handled_and_deleted(client, staging):
    response = _upload(client, "broken.pdf", PDF_BYTES, "application/pdf")
    assert response.status_code == 200
    state = _wait_for(client, response.json()["session_id"], {"paused", "completed", "error"})
    assert state["status"] == "paused"
    assert state["invoice_state"]["decision_trail"][0]["output_summary"]["extraction"] == "error"
    assert os.listdir(staging) == []


def test_the_file_is_deleted_even_when_the_workflow_itself_fails(client, staging, monkeypatch):
    async def exploding_run_async(*args, **kwargs):
        raise RuntimeError("workflow exploded")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(server.runner, "run_async", exploding_run_async)
    response = _upload(client, "a.pdf", PDF_BYTES, "application/pdf")
    assert response.status_code == 200
    state = _wait_for(client, response.json()["session_id"], {"error"})
    assert state["error_message"] == "workflow exploded"
    assert os.listdir(staging) == []
    assert server._documents_in_flight == 0


def test_the_file_is_deleted_when_the_run_cannot_start(client, staging, monkeypatch):
    async def no_session(*args, **kwargs):
        raise RuntimeError("session store down")

    monkeypatch.setattr(server.runner.session_service, "create_session", no_session)
    with pytest.raises(RuntimeError, match="session store down"):
        _upload(client, "a.pdf", PDF_BYTES, "application/pdf")
    assert os.listdir(staging) == []
    assert server._documents_in_flight == 0


def test_two_runs_of_the_same_document_never_share_a_session(client, staging, monkeypatch):
    monkeypatch.setenv("AP_MAX_CONCURRENT_UPLOADS", "5")
    ids = [_upload(client, "same-name.pdf", PDF_BYTES, "application/pdf").json()["session_id"] for _ in range(3)]
    assert len(set(ids)) == 3
    assert all(re.search(r"_[0-9a-f]{8}$", i) for i in ids)
    for session_id in ids:
        _wait_for(client, session_id, {"paused", "completed", "error"})
    assert os.listdir(staging) == []


# --- the per-request guards ---------------------------------------------------------------------------------------------


@needs_pdftotext
def test_one_document_drives_at_most_one_extraction_and_one_gl_call(client, staging, monkeypatch):
    """The path has no loop: reader -> one extract call -> one GL call. A counting provider pins that."""
    calls = []

    class CountingProvider(FixtureProvider):
        def complete(self, task, prompt, doc_id):
            calls.append((task, doc_id))
            return super().complete(task, prompt, doc_id)

    counting = CountingProvider(FIXTURES)
    monkeypatch.setattr("ap_invoice_processor.document_intake.get_provider", lambda: counting)
    monkeypatch.setattr("ap_invoice_processor.nodes.get_provider", lambda: counting)

    response = _upload(client, "en-001.pdf", _corpus_bytes("pdf", "en-001.pdf"), "application/pdf")
    state = _wait_for(client, response.json()["session_id"], {"paused", "completed", "error"})
    assert state["status"] == "paused"
    assert calls == [("extract", "en-001"), ("gl", "en-001")]

    client.post(f"/api/sessions/{response.json()['session_id']}/triage", json={"decision": "rejected", "reasoning": "no"})
    _wait_for(client, response.json()["session_id"], {"aborted", "completed", "error"})
    assert len(calls) == 2, "the resume leg must not call the provider again"


def test_a_busy_instance_refuses_another_document_with_429_and_frees_the_slot_afterwards(client, staging, monkeypatch):
    monkeypatch.setenv("AP_MAX_CONCURRENT_UPLOADS", "1")
    release = threading.Event()

    async def held_leg(session_id, adk_session_id, new_msg=None):
        # The leg runs on a worker thread's own loop, so it waits on a thread-safe flag.
        while not release.is_set():
            await asyncio.sleep(0.01)

    monkeypatch.setattr(server, "_run_workflow_leg", held_leg)
    first = _upload(client, "a.pdf", PDF_BYTES, "application/pdf")
    assert first.status_code == 200
    assert server._documents_in_flight == 1
    assert len(os.listdir(staging)) == 1

    second = _upload(client, "b.pdf", PDF_BYTES, "application/pdf")
    assert second.status_code == 429
    sample = client.post("/api/run-sample", json={"sample_id": "en-001.pdf"})
    assert sample.status_code == 429
    assert len(os.listdir(staging)) == 1, "a refused document must not stay on disk"

    release.set()
    deadline = time.time() + 10
    while server._documents_in_flight and time.time() < deadline:
        time.sleep(0.05)
    assert server._documents_in_flight == 0
    assert os.listdir(staging) == []
    assert _upload(client, "c.pdf", PDF_BYTES, "application/pdf").status_code == 200


def test_a_bad_concurrency_setting_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("AP_MAX_CONCURRENT_UPLOADS", "lots")
    assert server._max_concurrent_documents() == server.DEFAULT_MAX_CONCURRENT_DOCUMENTS
    monkeypatch.setenv("AP_MAX_CONCURRENT_UPLOADS", "0")
    assert server._max_concurrent_documents() == 1


# --- key handling -----------------------------------------------------------------------------------------------------


@needs_pdftotext
def test_the_api_key_never_appears_in_responses_logs_or_output_even_when_the_api_echoes_it(
    client, staging, monkeypatch, tmp_path, caplog, capsys
):
    """A live-provider upload where the API's error body repeats the key: nothing the dashboard returns or prints holds it."""
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("AP_LLM_SPEND_CAP_USD", "2")
    monkeypatch.setenv("AP_SPEND_LEDGER_PATH", str(tmp_path / "ledger" / "spend.json"))

    def echoing_api(request: httpx2.Request) -> httpx2.Response:
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": f"bad key {SENTINEL_KEY}"}}
        return httpx2.Response(400, json=body)

    real_build = anthropic_provider.build_anthropic_provider

    def build_with_mock_transport(max_usd=None, **kwargs):
        return real_build(max_usd=max_usd, http_client=httpx2.Client(transport=httpx2.MockTransport(echoing_api)), **kwargs)

    monkeypatch.setattr(anthropic_provider, "build_anthropic_provider", build_with_mock_transport)

    with caplog.at_level(logging.DEBUG):
        response = _upload(client, "en-001.pdf", _corpus_bytes("pdf", "en-001.pdf"), "application/pdf")
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        state = _wait_for(client, session_id, {"paused", "completed", "error"})
        assert state["status"] == "paused"
        assert state["invoice_state"]["decision_trail"][0]["output_summary"]["extraction"] == "error"
        everything = json.dumps(state) + response.text + client.get("/api/config").text + caplog.text
    captured = capsys.readouterr()
    assert SENTINEL_KEY not in everything + captured.out + captured.err
    assert (tmp_path / "ledger" / "spend.json").exists(), "the ledger path from AP_SPEND_LEDGER_PATH was not used"
    assert os.listdir(staging) == []


def test_a_missing_key_is_reported_without_echoing_any_secret(client, staging, monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AP_LLM_SPEND_CAP_USD", "2")
    response = _upload(client, "en-001.pdf", _corpus_bytes("pdf", "en-001.pdf"), "application/pdf")
    assert response.status_code == 200
    state = _wait_for(client, response.json()["session_id"], {"error", "paused", "completed"})
    assert state["status"] == "error"
    assert "ANTHROPIC_API_KEY is not set" in state["error_message"]
    assert os.listdir(staging) == []


# --- document limits ---------------------------------------------------------------------------------------------------

needs_pdfinfo = pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo not installed")


def _no_staged_files_or_slots(staging):
    assert os.listdir(staging) == []
    assert server._documents_in_flight == 0


def test_an_upload_over_the_default_page_cap_is_refused_with_a_clear_message_and_leaves_nothing(client, staging, monkeypatch):
    monkeypatch.setattr(uploads, "pdf_info", lambda path, last_page: reader.PdfInfo(pages=11, longest_edge_pts=792))
    response = _upload(client, "big.pdf", PDF_BYTES, "application/pdf")
    assert response.status_code == 413
    assert "11 pages" in response.json()["detail"] and "10" in response.json()["detail"]
    _no_staged_files_or_slots(staging)


def test_a_sample_goes_through_the_same_page_check(client, staging, monkeypatch):
    monkeypatch.setattr(uploads, "pdf_info", lambda path, last_page: reader.PdfInfo(pages=99, longest_edge_pts=792))
    response = client.post("/api/run-sample", json={"sample_id": "en-001.pdf"})
    assert response.status_code == 413
    _no_staged_files_or_slots(staging)


def test_a_refused_page_count_does_not_use_up_a_slot(client, staging, monkeypatch):
    monkeypatch.setenv("AP_MAX_CONCURRENT_UPLOADS", "1")
    monkeypatch.setattr(uploads, "pdf_info", lambda path, last_page: reader.PdfInfo(pages=11, longest_edge_pts=792))
    for _ in range(3):
        assert _upload(client, "big.pdf", PDF_BYTES, "application/pdf").status_code == 413
    assert server._documents_in_flight == 0


@needs_pdfinfo
def test_a_real_eleven_page_pdf_is_refused_and_ten_pages_are_accepted(client, staging):
    refused = _upload(client, "eleven.pdf", make_pdf(pages=11), "application/pdf")
    assert refused.status_code == 413 and "11 pages" in refused.json()["detail"]
    _no_staged_files_or_slots(staging)
    accepted = _upload(client, "ten.pdf", make_pdf(pages=10), "application/pdf")
    assert accepted.status_code == 200
    _wait_for(client, accepted.json()["session_id"], {"paused", "completed", "error"})


@needs_pdfinfo
def test_the_page_cap_follows_the_environment_in_the_web_layer(client, staging, monkeypatch):
    monkeypatch.setenv("AP_MAX_PDF_PAGES", "2")
    assert _upload(client, "three.pdf", make_pdf(pages=3), "application/pdf").status_code == 413
    ok = _upload(client, "two.pdf", make_pdf(pages=2), "application/pdf")
    assert ok.status_code == 200
    _wait_for(client, ok.json()["session_id"], {"paused", "completed", "error"})


def test_an_image_over_forty_megapixels_is_refused_before_any_staging(client, staging):
    response = _upload(client, "huge.png", png_header(10_000, 5_000), "image/png")
    assert response.status_code == 413
    assert "megapixel" in response.json()["detail"]
    _no_staged_files_or_slots(staging)


# --- the event loop stays free while a document is processed -------------------------------------------------------------


def test_polling_stays_responsive_while_a_slow_document_run_blocks_its_own_thread(client, staging, monkeypatch):
    """A sync workflow node (OCR, the model call) blocks whatever thread runs it; that must not be the server's loop."""
    started = threading.Event()
    finished = threading.Event()
    seen = {}

    async def blocking_leg(session_id, adk_session_id, new_msg=None):
        seen["thread"] = threading.get_ident()
        started.set()
        time.sleep(2.5)  # blocks this thread's loop, as the reader's subprocess calls do
        finished.set()

    monkeypatch.setattr(server, "_run_workflow_leg", blocking_leg)
    server_thread = client.portal.call(threading.get_ident)
    response = _upload(client, "slow.pdf", PDF_BYTES, "application/pdf")
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert started.wait(5)

    began = time.monotonic()
    poll = client.get(f"/api/sessions/{session_id}")
    other = client.get("/api/config")
    elapsed = time.monotonic() - began
    assert poll.status_code == 200 and other.status_code == 200
    assert not finished.is_set(), "the run ended before the poll was answered, so this proved nothing"
    assert elapsed < 1.0, f"polling took {elapsed:.2f}s while a document run was blocked"
    assert seen["thread"] != server_thread, "the workflow ran on the server's event loop thread"

    assert finished.wait(10)
    deadline = time.time() + 10
    while server._documents_in_flight and time.time() < deadline:
        time.sleep(0.05)
    _no_staged_files_or_slots(staging)


def test_the_slot_and_file_are_released_when_a_worker_thread_leg_raises(client, staging, monkeypatch):
    async def exploding_leg(session_id, adk_session_id, new_msg=None):
        raise RuntimeError("worker blew up")

    monkeypatch.setattr(server, "_run_workflow_leg", exploding_leg)
    assert _upload(client, "boom.pdf", PDF_BYTES, "application/pdf").status_code == 200
    deadline = time.time() + 10
    while server._documents_in_flight and time.time() < deadline:
        time.sleep(0.05)
    _no_staged_files_or_slots(staging)


def _blocked_while_session_lock_is_held(action):
    """True if `action` (run on another thread) waits for the session lock and finishes once it is released."""
    done = threading.Event()
    worker = threading.Thread(target=lambda: (action(), done.set()))
    with server._SESSION_LOCK:
        worker.start()
        blocked = not done.wait(0.3)
    worker.join(5)
    return blocked and done.is_set()


def test_session_state_is_read_under_the_session_lock(client, monkeypatch):
    monkeypatch.setitem(server.ACTIVE_SESSIONS, "sess_lock", {"session_id": "sess_lock", "status": "running"})
    assert _blocked_while_session_lock_is_held(lambda: client.get("/api/sessions/sess_lock"))


def test_session_state_is_written_under_the_session_lock():
    session = {"status": "running", "is_paused_at_gate": False}
    assert _blocked_while_session_lock_is_held(lambda: server._update_session(session, status="paused", is_paused_at_gate=True))
    assert session == {"status": "paused", "is_paused_at_gate": True}

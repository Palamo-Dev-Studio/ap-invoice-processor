import os
import copy
import json
import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from ap_invoice_processor.graph import root_agent
from ap_invoice_processor.llm.provider import DEFAULT_PROVIDER, PROVIDER_ENV
from ap_invoice_processor.hitl import (
    HUMAN_GATE_INTERRUPT_ID,
    is_paused_at_gate,
    build_resume_message,
    poster_ran,
)
from web.presentation import DEMO_NOTICE, demo_mode_enabled, node_subtitles, render_index
from web.uploads import (
    ALLOWED_TYPES,
    MAX_UPLOAD_BYTES,
    StagedDocument,
    UploadRejected,
    check_pdf_page_limit,
    list_samples,
    remove_staged,
    stage_document,
    stage_sample,
)

app_instance = App(name="ap_copilot_app", root_agent=root_agent)
runner = InMemoryRunner(app=app_instance)

web_app = FastAPI(title="AP Copilot - Autonomous Invoice Processing Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
# Headroom over MAX_UPLOAD_BYTES for the multipart envelope when the Content-Length header is checked up front.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_CONCURRENT_ENV = "AP_MAX_CONCURRENT_UPLOADS"
DEFAULT_MAX_CONCURRENT_DOCUMENTS = 2

web_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}
# A workflow leg runs on a worker thread (see _run_leg_in_worker_thread), so a session dict is written there and read
# on the event loop thread. Every write after a session's creation and every copy for a reader holds this lock, which
# makes a group of fields change together as far as any reader can tell.
_SESSION_LOCK = threading.Lock()
# Where workflow legs run. The reader's pdftoppm/tesseract calls and the synchronous model client block their thread
# for as long as they take (minutes for a long scan), so they must not share the thread that serves requests.
_WORKFLOW_THREADS = 16
_WORKFLOW_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKFLOW_THREADS, thread_name_prefix="ap-workflow")
# Strong references to running workflow tasks: the event loop keeps only weak ones, and a task collected mid-run would
# skip the cleanup in _execute_workflow's finally block.
_BACKGROUND_TASKS: set = set()
# Document runs (uploads and samples) whose first workflow leg has not finished. Read and changed only on the event
# loop thread with no await in between, so the limit check below cannot race.
_documents_in_flight = 0


def _max_concurrent_documents() -> int:
    try:
        return max(1, int(os.environ.get(MAX_CONCURRENT_ENV, DEFAULT_MAX_CONCURRENT_DOCUMENTS)))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_DOCUMENTS


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _update_session(sess_info: Dict[str, Any], **fields: Any) -> None:
    """Change several fields of one session dict as a single step, from any thread."""
    with _SESSION_LOCK:
        sess_info.update(fields)


def _claim_document_slot() -> bool:
    global _documents_in_flight
    if _documents_in_flight >= _max_concurrent_documents():
        return False
    _documents_in_flight += 1
    return True


def _finish_document_run(staged_dir: Optional[str]) -> None:
    """End a document run: delete its staged file and free its slot. A run without a staged dir is a no-op."""
    global _documents_in_flight
    if staged_dir is None:
        return
    remove_staged(staged_dir)
    _documents_in_flight = max(0, _documents_in_flight - 1)


def load_synthetic_invoices() -> list:
    inv_path = os.path.join(DATA_DIR, "synthetic_invoices", "invoices.json")
    if os.path.exists(inv_path):
        with open(inv_path, "r") as f:
            return json.load(f)
    return []

@web_app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=render_index(f.read(), demo_mode_enabled()))

@web_app.get("/api/config")
async def get_config():
    demo = demo_mode_enabled()
    return {
        "demo_mode": demo,
        "notice": DEMO_NOTICE if demo else None,
        "llm_provider": (os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower(),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "accepted_types": sorted(ALLOWED_TYPES),
    }

@web_app.get("/api/samples")
async def get_samples():
    return list_samples(CORPUS_DIR)

@web_app.get("/api/invoices")
async def list_invoices():
    return load_synthetic_invoices()

@web_app.get("/api/sessions/{session_id}")
async def get_session_state(session_id: str):
    if session_id not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    with _SESSION_LOCK:
        session = dict(ACTIVE_SESSIONS[session_id])
    return {**session, "node_subtitles": node_subtitles(session.get("invoice_state"))}

class RunInvoiceRequest(BaseModel):
    invoice_id: str

class RunSampleRequest(BaseModel):
    sample_id: str

class RunCustomRequest(BaseModel):
    vendor_name: str
    total_amount: float
    po_number: Optional[str] = None
    line_item_description: str

class HumanTriageRequest(BaseModel):
    decision: str  # "approved" or "rejected"
    reasoning: Optional[str] = None


async def _start_invoice_run(invoice: dict, staged_dir: Optional[str] = None) -> str:
    """Create a fresh ADK session for an invoice payload, register it in
    ACTIVE_SESSIONS, and kick off _execute_workflow. Shared by the pre-baked
    (/api/run), custom (/api/run-custom) and document (/api/run-upload,
    /api/run-sample) entry points so they run through the identical session +
    workflow machinery. Returns the session id.

    `staged_dir` is the temp dir of a document run's file: it is deleted, and the
    run's slot freed, when the first workflow leg ends, or here if the run cannot start."""
    try:
        return await _launch_invoice_run(invoice, staged_dir)
    except BaseException:
        _finish_document_run(staged_dir)
        raise


async def _launch_invoice_run(invoice: dict, staged_dir: Optional[str]) -> str:
    invoice_id = invoice["id"]
    # The suffix keeps two runs of the same invoice in the same second from sharing a session.
    session_id = f"sess_{invoice_id}_{int(asyncio.get_event_loop().time())}_{uuid.uuid4().hex[:8]}"

    adk_session = await runner.session_service.create_session(
        app_name="ap_copilot_app", user_id="demo_user"
    )

    ACTIVE_SESSIONS[session_id] = {
        "session_id": session_id,
        "adk_session_id": adk_session.id,
        "invoice_id": invoice_id,
        "status": "running",
        "current_node": "START",
        "invoice_state": None,
        "is_paused_at_gate": False,
        "interrupt_id": None
    }

    input_text = json.dumps(invoice)
    start_msg = types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
    _spawn(_execute_workflow(session_id, adk_session.id, new_msg=start_msg, staged_dir=staged_dir))
    return session_id


def _build_custom_invoice(req: RunCustomRequest) -> dict:
    """Synthesize a full invoice payload from simple tester inputs, matching the
    exact shape intake_node expects (id + raw_text + simulated_extraction block).
    Confidences are set HIGH (~0.97) so the low-confidence rail only fires when an
    input legitimately trips a different policy check (ceiling, vendor, PO)."""
    invoice_id = f"CUSTOM-{int(time.time())}"
    amount = round(float(req.total_amount), 2)
    po_number = req.po_number.strip() if req.po_number and req.po_number.strip() else None
    date = datetime.now().strftime("%Y-%m-%d")

    raw_text = (
        f"INVOICE #{invoice_id}\n"
        f"Vendor: {req.vendor_name}\n"
        f"Date: {date}\n"
        f"PO: {po_number or 'N/A'}\n"
        f"Total: ${amount:.2f}"
    )

    line_item = {
        "description": req.line_item_description,
        "qty": 1,
        "unit_price": amount,
        "amount": amount,
    }

    return {
        "id": invoice_id,
        "test_case_type": "custom",
        "description": "Tester-submitted custom invoice",
        "raw_text": raw_text,
        "simulated_extraction": {
            "vendor_name": req.vendor_name,
            "invoice_number": invoice_id,
            "date": date,
            "po_number": po_number,
            "total_amount": amount,
            "line_items": [line_item],
            "confidence": {
                "vendor_name": 0.97,
                "invoice_number": 0.97,
                "date": 0.97,
                "total_amount": 0.97,
                "line_items": 0.97,
            },
        },
    }


@web_app.post("/api/run")
async def run_invoice(req: RunInvoiceRequest):
    invoices = load_synthetic_invoices()
    selected = next((inv for inv in invoices if inv["id"] == req.invoice_id), None)
    if not selected:
        raise HTTPException(status_code=404, detail=f"Invoice {req.invoice_id} not found")

    session_id = await _start_invoice_run(selected)
    return {"session_id": session_id, "status": "started"}


@web_app.post("/api/run-custom")
async def run_custom_invoice(req: RunCustomRequest):
    invoice = _build_custom_invoice(req)
    session_id = await _start_invoice_run(invoice)
    return {"session_id": session_id, "status": "started", "invoice_id": invoice["id"]}

async def _start_document_run(staged: StagedDocument) -> dict:
    """Start one workflow run on one staged document, or refuse it: 413 over the page cap, 429 when the instance is busy."""
    if staged.path.lower().endswith(".pdf"):
        try:
            # pdfinfo is a subprocess, so it runs on a worker thread rather than on the event loop.
            await asyncio.to_thread(check_pdf_page_limit, staged.path)
        except UploadRejected as exc:
            remove_staged(staged.directory)
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        except BaseException:
            remove_staged(staged.directory)
            raise
    if not _claim_document_slot():
        remove_staged(staged.directory)
        raise HTTPException(status_code=429, detail="Another document is still being processed. Try again in a moment.")
    # The payload names the file stem as the id, so the invoice id and document id are the same readable name.
    session_id = await _start_invoice_run({"id": staged.stem, "document_path": staged.path}, staged_dir=staged.directory)
    return {"session_id": session_id, "status": "started", "invoice_id": staged.stem}


@web_app.post("/api/run-upload")
async def run_upload(request: Request):
    """Run the document path on one uploaded PDF/PNG/JPEG. The request must carry exactly one file part named
    "file"; the file is staged in a private temp dir and deleted when the workflow's first leg ends."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than the 5 MB limit.")
    # The multipart parser buffers the body before this handler sees it, so a client that omits Content-Length is
    # bounded by the platform's request-size limit (Cloud Run: 32 MiB), not by this check; the bytes read below are capped.
    async with request.form(max_files=1, max_fields=1) as form:
        parts = form.multi_items()
        if len(parts) != 1 or parts[0][0] != "file" or not hasattr(parts[0][1], "read"):
            raise HTTPException(status_code=400, detail="Send exactly one document, in a form field named 'file'.")
        upload = parts[0][1]
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        filename = upload.filename
    try:
        staged = stage_document(filename, data)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    return await _start_document_run(staged)


@web_app.post("/api/run-sample")
async def run_sample(req: RunSampleRequest):
    try:
        staged = stage_sample(CORPUS_DIR, req.sample_id)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    return await _start_document_run(staged)


async def _execute_workflow(session_id: str, adk_session_id: str, new_msg: types.Content = None, staged_dir: Optional[str] = None):
    try:
        # The leg runs on a worker thread; this coroutine stays on the event loop and only waits for it, so the slot
        # accounting and the file cleanup below still run on the loop thread.
        await asyncio.get_running_loop().run_in_executor(
            _WORKFLOW_EXECUTOR, _run_leg_in_worker_thread, session_id, adk_session_id, new_msg
        )
    finally:
        # The document is read only by the Intake node, in the first leg, so it is deleted once that leg ends
        # (completed, paused at the gate, or failed). The resume leg after a triage decision never reads it.
        _finish_document_run(staged_dir)


def _run_leg_in_worker_thread(session_id: str, adk_session_id: str, new_msg: Optional[types.Content]) -> None:
    """Drive one workflow leg to its end on this thread, in an event loop of its own."""
    asyncio.run(_run_workflow_leg(session_id, adk_session_id, new_msg))


async def _run_workflow_leg(session_id: str, adk_session_id: str, new_msg: types.Content = None):
    """Run one leg of a session's workflow. This executes on a worker thread, so it changes the session only through
    _update_session and publishes an invoice state as a private copy."""
    sess_info = ACTIVE_SESSIONS.get(session_id)
    if not sess_info:
        return

    try:
        paused = False
        # Consume the stream fully; the runner suspends at the gate on its own (the
        # gate event is the last one yielded). Returning mid-stream cancels the
        # workflow and emits noisy errors.
        async for event in runner.run_async(
            user_id="demo_user",
            session_id=adk_session_id,
            new_message=new_msg,
        ):
            if is_paused_at_gate(event):
                paused = True

            if event.output and isinstance(event.output, dict) and "invoice_id" in event.output:
                st = copy.deepcopy(event.output)
                update: Dict[str, Any] = {"invoice_state": st}
                if st.get("decision_trail"):
                    last_step = st["decision_trail"][-1]
                    update["current_node"] = last_step["node_name"]
                _update_session(sess_info, **update)

        if paused:
            _update_session(
                sess_info,
                status="paused",
                is_paused_at_gate=True,
                interrupt_id=HUMAN_GATE_INTERRUPT_ID,
                current_node="Human Gate",
            )
            return

        # The stream ended without pausing: the Poster ran (posted or aborted).
        final_state = sess_info.get("invoice_state") or {}
        if isinstance(final_state, dict) and final_state.get("human_decision") == "rejected" and poster_ran(final_state):
            status = "aborted"
        else:
            status = "completed"
        _update_session(sess_info, status=status, is_paused_at_gate=False, current_node="Poster")
    except Exception as e:
        print(f"Workflow Execution Error for {session_id}: {e}")
        _update_session(sess_info, status="error", error_message=str(e))

@web_app.post("/api/sessions/{session_id}/triage")
async def submit_triage(session_id: str, req: HumanTriageRequest):
    sess_info = ACTIVE_SESSIONS.get(session_id)
    if not sess_info:
        raise HTTPException(status_code=404, detail="Session not found")
    if not sess_info.get("is_paused_at_gate"):
        raise HTTPException(status_code=400, detail="Session is not paused at Human Gate")

    decision = req.decision.lower()
    reasoning = req.reasoning or f"Reviewed and {decision} via dashboard triage."

    _update_session(sess_info, status="resuming", is_paused_at_gate=False)

    resume_msg = build_resume_message(decision, reasoning)
    _spawn(_execute_workflow(session_id, sess_info["adk_session_id"], new_msg=resume_msg))
    return {"status": "resumed", "decision": decision}

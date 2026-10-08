# ABOUTME: Tests for the live Anthropic provider with the HTTP layer mocked: request shape, structured output, errors, cap.
# ABOUTME: The real SDK client runs against an httpx2 MockTransport, so no test here touches the network or a real key.
import json
import os
import shutil
import threading
import time
from decimal import Decimal

import anthropic
import httpx2
import pytest

from ap_invoice_processor.document_intake import fill_state_from_document
from ap_invoice_processor.llm import anthropic_provider as ap
from ap_invoice_processor.llm import spend as spend_module
from ap_invoice_processor.llm.anthropic_provider import AnthropicProvider, build_anthropic_provider
from ap_invoice_processor.llm.extraction import ExtractionError, extract_invoice
from ap_invoice_processor.llm.extraction import ExtractedInvoice
from ap_invoice_processor.llm.gl import code_lines, load_chart
from ap_invoice_processor.llm.provider import LLMProvider, LLMProviderError, ProviderConfigError, get_provider
from ap_invoice_processor.llm.spend import SpendCapExceeded, SpendLedgerError, SpendTracker
from ap_invoice_processor.models import InvoiceState
from ap_invoice_processor.reader import ReaderOutput

D = Decimal
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN_001_PDF = os.path.join(ROOT, "data", "corpus", "pdf", "en-001.pdf")
KEY = "sk-ant-TEST-KEY-do-not-leak-0123456789"
INVOICE = {
    "vendor_name": "Acme Marketing Solutions",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-07-01",
    "due_date": "2026-07-31",
    "currency": "USD",
    "po_number": None,
    "subtotal": 100.0,
    "tax": 8.0,
    "total": 108.0,
    "line_items": [{"description": "Consulting", "quantity": 2, "unit_price": 50.0, "amount": 100.0}],
}


def message(text=None, stop_reason="end_turn", usage=None, content=None, stop_details=None):
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 500},
    }
    if stop_details:
        body["stop_details"] = stop_details
    return httpx2.Response(200, json=body)


def api_error(status, kind="api_error", text="boom"):
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": text}})


class Transport:
    """Serves scripted responses in order (the last one repeats) and records every request that reaches it."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def make_provider(tmp_path, transport, cap="10", run_cap=None, max_retries=2, **kwargs):
    client = anthropic.Anthropic(
        api_key=KEY, http_client=httpx2.Client(transport=httpx2.MockTransport(transport)), max_retries=max_retries
    )
    tracker = SpendTracker(str(tmp_path / "spend.json"), cap_usd=D(cap), run_cap_usd=None if run_cap is None else D(run_cap))
    return AnthropicProvider(client, tracker, **kwargs)


def estimate(prompt, max_tokens=ap.DEFAULT_MAX_TOKENS, task="extract"):
    """What the provider estimates for one attempt of `prompt`; the same arithmetic complete() uses."""
    return ap.estimate_cost_usd(
        ap.DEFAULT_MODEL, prompt, max_tokens,
        extra_input_chars=len(ap.SYSTEM_PROMPT) + len(json.dumps(ap.TASK_SCHEMAS[task])),
    )


def hold(tmp_path, prompt, max_retries=2, **kwargs):
    """The sum reserved for one call: one estimate per possible attempt (the first plus every SDK retry)."""
    return estimate(prompt, **kwargs) * (max_retries + 1)


def ledger_total(tmp_path):
    return D(json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["total_usd"])


# --- the request ------------------------------------------------------------------------------------------------------


def test_the_provider_satisfies_the_protocol(tmp_path):
    assert isinstance(make_provider(tmp_path, Transport(message("{}"))), LLMProvider)


def test_an_extract_request_asks_haiku_55_for_schema_constrained_json_with_an_explicit_effort(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    out = make_provider(tmp_path, t).complete("extract", "PROMPT TEXT", "doc-1")
    assert out == INVOICE
    (body,) = t.bodies
    assert body["model"] == "claude-haiku-5-5"
    assert body["messages"] == [{"role": "user", "content": "PROMPT TEXT"}]
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": ap.EXTRACT_SCHEMA}
    assert body["max_tokens"] == ap.DEFAULT_MAX_TOKENS
    assert t.requests[0].headers["x-api-key"] == KEY


def test_the_request_carries_none_of_the_parameters_haiku_55_rejects_or_that_would_invent_output(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    (body,) = t.bodies
    for rejected in ("temperature", "top_p", "top_k", "tool_choice", "tools", "thinking", "stream"):
        assert rejected not in body
    assert body["messages"][-1]["role"] == "user"  # no assistant prefill


def test_the_system_prompt_is_static_and_treats_the_document_as_untrusted(tmp_path):
    t = Transport(message(json.dumps(INVOICE)), message(json.dumps({"lines": []})))
    p = make_provider(tmp_path, t)
    p.complete("extract", "first document", "doc-1")
    p.complete("gl", "second prompt", "doc-1")
    first, second = t.bodies
    assert first["system"] == second["system"] == ap.SYSTEM_PROMPT
    assert "untrusted" in ap.SYSTEM_PROMPT and "instructions" in ap.SYSTEM_PROMPT


def test_no_cache_control_is_sent(tmp_path):
    # Prompt caching is deliberately off: the static prefix sits near Haiku 5.5's 512-token minimum, the document text
    # follows it in the same prompt string, and the saving would be a few micro-dollars per call.
    t = Transport(message(json.dumps(INVOICE)))
    make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    assert "cache_control" not in t.requests[0].content.decode("utf-8")


def test_the_effort_and_output_ceiling_are_configurable(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    make_provider(tmp_path, t, effort="low", max_tokens=2048).complete("extract", "p", "doc-1")
    (body,) = t.bodies
    assert body["output_config"]["effort"] == "low" and body["max_tokens"] == 2048


def test_an_unknown_task_is_an_error_and_sends_nothing(tmp_path):
    t = Transport(message("{}"))
    with pytest.raises(LLMProviderError, match="task"):
        make_provider(tmp_path, t).complete("summarise", "p", "doc-1")
    assert t.requests == []


# --- the structured-output schemas -----------------------------------------------------------------------------------


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _objects(v)


@pytest.mark.parametrize("schema", [ap.EXTRACT_SCHEMA, ap.GL_SCHEMA], ids=["extract", "gl"])
def test_schemas_use_only_what_structured_outputs_accepts(schema):
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    # transform_schema drops or rewrites anything the API does not support; an unchanged schema means nothing was lost.
    assert anthropic.transform_schema(schema) == schema


def test_the_extract_schema_names_exactly_the_fields_extraction_validates():
    assert set(ap.EXTRACT_SCHEMA["properties"]) == set(ExtractedInvoice.model_fields)
    line_props = ap.EXTRACT_SCHEMA["properties"]["line_items"]["items"]["properties"]
    from ap_invoice_processor.llm.extraction import ExtractedLineItem

    assert set(line_props) == set(ExtractedLineItem.model_fields)


def test_the_extract_schema_lets_the_model_say_null_for_every_header_field_including_the_total():
    for name, prop in ap.EXTRACT_SCHEMA["properties"].items():
        if name != "line_items":
            assert {"type": "null"} in prop["anyOf"], name


def test_the_gl_schema_matches_the_response_shape_the_coder_reads():
    line = ap.GL_SCHEMA["properties"]["lines"]["items"]["properties"]
    assert set(line) == {"line", "account", "confidence", "reason"}
    assert line["line"]["type"] == "integer" and line["account"]["type"] == "string"


# --- the response ------------------------------------------------------------------------------------------------------


def test_json_in_a_text_block_after_thinking_blocks_is_returned(tmp_path):
    content = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": json.dumps(INVOICE)},
    ]
    assert make_provider(tmp_path, Transport(message(content=content))).complete("extract", "p", "d") == INVOICE


def test_every_response_is_priced_from_its_usage_and_added_to_the_ledger(tmp_path):
    usage = {"input_tokens": 2000, "output_tokens": 1000, "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 4000}
    t = Transport(message(json.dumps(INVOICE), usage=usage))
    p = make_provider(tmp_path, t)
    p.complete("extract", "p", "doc-1")
    expected = (D(2000) * D("0.10") + D(1000) * D("0.50") + D(3000) * D("0.01") + D(4000) * D("0.125")) / D(1_000_000)
    assert ledger_total(tmp_path) == expected
    assert p.tracker.run_spent_usd == expected
    p.complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == expected * 2


def test_spend_persists_across_provider_instances(tmp_path):
    for _ in range(2):
        make_provider(tmp_path, Transport(message(json.dumps(INVOICE)))).complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == D(2) * (D(1000) * D("0.10") + D(500) * D("0.50")) / D(1_000_000)


# --- failures never produce values -------------------------------------------------------------------------------------


def test_a_refusal_is_a_provider_error_naming_the_category_and_is_still_billed(tmp_path):
    t = Transport(message(content=[], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "x"}))
    with pytest.raises(LLMProviderError, match="declined.*cyber"):
        make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    assert len(t.requests) == 1  # a refusal is not retried
    assert ledger_total(tmp_path) > 0


def test_a_truncated_response_is_an_error_not_a_partial_value(tmp_path):
    t = Transport(message('{"vendor_name": "Acme', stop_reason="max_tokens"))
    with pytest.raises(LLMProviderError, match="max_tokens"):
        make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) > 0


@pytest.mark.parametrize(
    "text, fragment",
    [("not json at all", "not valid JSON"), ("[1, 2]", "JSON object"), ('"just a string"', "JSON object"), ("", "no text")],
)
def test_unusable_model_text_is_an_error(tmp_path, text, fragment):
    with pytest.raises(LLMProviderError, match=fragment):
        make_provider(tmp_path, Transport(message(text))).complete("extract", "p", "doc-1")


def test_a_response_without_a_text_block_is_an_error(tmp_path):
    content = [{"type": "thinking", "thinking": "", "signature": "s"}]
    with pytest.raises(LLMProviderError, match="no text"):
        make_provider(tmp_path, Transport(message(content=content))).complete("extract", "p", "doc-1")


@pytest.mark.parametrize("stop", ["pause_turn", "tool_use", "stop_sequence"])
def test_an_unexpected_stop_reason_is_an_error(tmp_path, stop):
    with pytest.raises(LLMProviderError, match="stop_reason"):
        make_provider(tmp_path, Transport(message(json.dumps(INVOICE), stop_reason=stop))).complete("extract", "p", "d")


@pytest.mark.parametrize("status, kind", [(400, "invalid_request_error"), (401, "authentication_error"), (403, "permission_error"), (404, "not_found_error")])
def test_client_errors_fail_at_once_without_retrying(tmp_path, status, kind):
    t = Transport(api_error(status, kind))
    with pytest.raises(LLMProviderError, match=str(status)):
        make_provider(tmp_path, t, max_retries=3).complete("extract", "p", "doc-1")
    assert len(t.requests) == 1
    # The provider does not assume a rejected request cost nothing: the reservation stays counted (see the
    # unknown-outcome tests below), so a failure can only ever over-count spend.
    assert ledger_total(tmp_path) == hold(tmp_path, "p", max_retries=3)


def test_a_rate_limit_is_retried_and_then_succeeds(tmp_path):
    t = Transport(api_error(429, "rate_limit_error"), message(json.dumps(INVOICE)))
    assert make_provider(tmp_path, t).complete("extract", "p", "doc-1") == INVOICE
    assert len(t.requests) == 2


def test_a_persistent_rate_limit_ends_as_a_provider_error_after_the_retry_budget(tmp_path):
    t = Transport(api_error(429, "rate_limit_error"))
    with pytest.raises(LLMProviderError, match="429"):
        make_provider(tmp_path, t, max_retries=2).complete("extract", "p", "doc-1")
    assert len(t.requests) == 3


@pytest.mark.parametrize("status", [500, 503, 529])
def test_server_errors_are_retried_then_surface_as_provider_errors(tmp_path, status):
    t = Transport(api_error(status))
    with pytest.raises(LLMProviderError, match=str(status)):
        make_provider(tmp_path, t, max_retries=1).complete("extract", "p", "doc-1")
    assert len(t.requests) == 2


def test_a_timeout_surfaces_as_a_provider_error(tmp_path):
    t = Transport(httpx2.ReadTimeout("slow"))
    with pytest.raises(LLMProviderError, match="timed out"):
        make_provider(tmp_path, t, max_retries=1).complete("extract", "p", "doc-1")
    assert len(t.requests) == 2


def test_a_connection_failure_surfaces_as_a_provider_error(tmp_path):
    t = Transport(httpx2.ConnectError("down"))
    with pytest.raises(LLMProviderError, match="connect"):
        make_provider(tmp_path, t, max_retries=0).complete("extract", "p", "doc-1")


def test_the_api_key_never_appears_in_an_error_even_if_the_server_echoes_it(tmp_path):
    t = Transport(api_error(401, "authentication_error", f"invalid x-api-key {KEY}"))
    with pytest.raises(LLMProviderError) as info:
        make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    assert KEY not in str(info.value) and "[redacted]" in str(info.value)


# --- the spend cap, applied before anything is sent ---------------------------------------------------------------------


def test_a_call_that_would_pass_the_cap_is_refused_before_any_request_is_sent(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    p = make_provider(tmp_path, t, cap="0.001")  # the 8,192-token output ceiling alone is worth more than this
    with pytest.raises(SpendCapExceeded):
        p.complete("extract", "p", "doc-1")
    assert t.requests == []
    assert p.cap_reached is True
    assert not (tmp_path / "spend.json").exists()


def test_a_call_that_fits_under_the_cap_is_sent_and_leaves_the_flag_unset(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    p = make_provider(tmp_path, t, cap="10")
    p.complete("extract", "p", "doc-1")
    assert len(t.requests) == 1 and p.cap_reached is False


def test_the_run_cap_stops_calls_while_the_global_cap_has_room(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    p = make_provider(tmp_path, t, cap="10", run_cap="0.015")
    p.complete("extract", "p", "doc-1")  # holds ~0.0125 (3 attempts x ~0.0042) while in flight, settles at ~0.00035
    for _ in range(20):
        try:
            p.complete("extract", "p", "doc-1")
        except SpendCapExceeded as exc:
            assert "run cap" in str(exc)
            break
    else:
        pytest.fail("the run cap never stopped the run")
    assert p.cap_reached is True
    assert p.tracker.run_spent_usd <= D("0.015")


def test_the_cap_is_checked_against_spend_from_earlier_runs(tmp_path):
    SpendTracker(str(tmp_path / "spend.json"), cap_usd=D("10")).record(D("9.999"))
    t = Transport(message(json.dumps(INVOICE)))
    with pytest.raises(SpendCapExceeded):
        make_provider(tmp_path, t, cap="10").complete("extract", "p", "doc-1")
    assert t.requests == []


def test_a_corrupt_ledger_blocks_the_call_before_it_is_sent(tmp_path):
    (tmp_path / "spend.json").write_text("garbage", encoding="utf-8")
    t = Transport(message(json.dumps(INVOICE)))
    with pytest.raises(LLMProviderError, match="unreadable"):
        make_provider(tmp_path, t).complete("extract", "p", "doc-1")
    assert t.requests == []


# --- reserve, send, settle ----------------------------------------------------------------------------------------------

ACTUAL = (D(1000) * D("0.10") + D(500) * D("0.50")) / D(1_000_000)  # the default message() usage


class Crash(BaseException):
    """Stands in for the process dying mid-call: not an Exception, so neither the SDK nor the provider catches it."""


def test_the_reservation_is_in_the_ledger_before_the_request_is_sent_and_covers_every_possible_attempt(tmp_path):
    seen = []

    def handler(request):
        seen.append(ledger_total(tmp_path))
        return message(json.dumps(INVOICE))

    make_provider(tmp_path, handler, max_retries=2).complete("extract", "PROMPT", "doc-1")
    # With the SDK allowed 2 retries, up to 3 requests can be billed, so 3 estimates are set aside up front.
    assert seen == [hold(tmp_path, "PROMPT", max_retries=2)] and seen[0] == estimate("PROMPT") * 3


def test_after_a_clean_first_attempt_the_reservation_is_replaced_by_the_observed_cost(tmp_path):
    p = make_provider(tmp_path, Transport(message(json.dumps(INVOICE))))
    p.complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == ACTUAL and p.tracker.reserved_usd == 0


@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_retries_that_preceded_a_success_are_charged_at_the_estimate_on_top_of_the_observed_cost(tmp_path, failed_attempts):
    # The earlier attempts' outcomes are unknown (a 429 or 5xx is usually not billed, a timeout may be), so each is
    # counted at the pre-call estimate, which is never below what one attempt can cost.
    t = Transport(*([api_error(500)] * failed_attempts), message(json.dumps(INVOICE)))
    p = make_provider(tmp_path, t, max_retries=2)
    p.complete("extract", "p", "doc-1")
    assert len(t.requests) == failed_attempts + 1
    assert ledger_total(tmp_path) == ACTUAL + estimate("p") * failed_attempts
    assert p.tracker.run_spent_usd == ledger_total(tmp_path)


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(httpx2.ReadTimeout("slow"), id="timeout"),
        pytest.param(httpx2.ConnectError("down"), id="connection error"),
        pytest.param(api_error(500), id="500"),
        pytest.param(api_error(529, "overloaded_error"), id="529"),
        pytest.param(api_error(429, "rate_limit_error"), id="429"),
        pytest.param(api_error(401, "authentication_error"), id="401"),
    ],
)
def test_a_call_whose_outcome_is_unknown_stays_counted_as_spent(tmp_path, outcome):
    # A timeout after the server accepted the request can still be billed, and a failure response does not prove it
    # was not; none of them leaves the ledger, so spend can only be over-counted, never missed.
    t = Transport(outcome)
    p = make_provider(tmp_path, t, max_retries=2)
    with pytest.raises(LLMProviderError):
        p.complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == hold(tmp_path, "p", max_retries=2)
    assert p.tracker.reserved_usd == ledger_total(tmp_path) and p.tracker.run_spent_usd == ledger_total(tmp_path)


def test_a_process_that_dies_mid_call_leaves_its_reservation_counted_for_the_next_run(tmp_path):
    def handler(request):
        raise Crash()

    p = make_provider(tmp_path, handler, max_retries=0)
    with pytest.raises(Crash):
        p.complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == estimate("p")
    # a later run sharing the ledger sees it: with a cap one estimate above the pending one, a second call no longer fits
    cap = format(estimate("p") * 2 - D("0.000001"), "f")
    with pytest.raises(SpendCapExceeded):
        make_provider(tmp_path, Transport(message(json.dumps(INVOICE))), cap=cap, max_retries=0).complete("extract", "p", "doc-1")


def test_unknown_outcomes_add_up_and_eventually_stop_the_run_at_the_cap(tmp_path):
    cap = format(hold(tmp_path, "p", max_retries=0) * 2, "f")
    t = Transport(httpx2.ConnectError("down"))
    p = make_provider(tmp_path, t, cap=cap, max_retries=0)
    for _ in range(2):
        with pytest.raises(LLMProviderError, match="connect"):
            p.complete("extract", "p", "doc-1")
    with pytest.raises(SpendCapExceeded):
        p.complete("extract", "p", "doc-1")
    assert len(t.requests) == 2 and p.cap_reached is True


def test_concurrent_calls_cannot_all_be_admitted_past_the_cap(tmp_path):
    cap = format(estimate("p") * 2, "f")  # room for exactly two calls in flight
    arrived, results = [], []
    release = threading.Event()

    def handler(request):
        arrived.append(1)
        release.wait(30)
        return message(json.dumps(INVOICE))

    def call():
        provider = make_provider(tmp_path, handler, cap=cap, max_retries=0)
        try:
            provider.complete("extract", "p", "doc-1")
            results.append("ok")
        except SpendCapExceeded:
            results.append("refused")

    threads = [threading.Thread(target=call) for _ in range(6)]
    for th in threads:
        th.start()
    deadline = time.monotonic() + 30
    while len(arrived) + results.count("refused") < 6 and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    threading.Event().wait(0.2)
    in_flight = len(arrived)
    release.set()
    for th in threads:
        th.join()
    assert in_flight == 2
    assert sorted(results) == ["ok", "ok", "refused", "refused", "refused", "refused"]
    assert ledger_total(tmp_path) == ACTUAL * 2


# --- ledger and cost failures after a billed call never escape the fallback paths --------------------------------------


def _ledger_write_fails_once_the_request_is_in(tmp_path, monkeypatch, patched):
    """A provider whose ledger can no longer be written by the time the (paid) response arrives."""

    def handler(request):
        patched.setattr(spend_module.os, "replace", _disk_full)
        return message(json.dumps(INVOICE))

    return make_provider(tmp_path, handler, max_retries=0)


def _disk_full(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_a_ledger_write_failure_after_a_billed_call_is_a_provider_error_and_keeps_the_reservation(tmp_path, monkeypatch):
    with monkeypatch.context() as patched:
        p = _ledger_write_fails_once_the_request_is_in(tmp_path, monkeypatch, patched)
        with pytest.raises(LLMProviderError, match="could not write spend ledger") as info:
            p.complete("extract", "p", "doc-1")
    assert isinstance(info.value, SpendLedgerError)
    assert ledger_total(tmp_path) == estimate("p")  # the settle never landed, so the reservation still counts
    assert p.tracker.run_spent_usd == estimate("p")


def test_a_ledger_that_cannot_be_written_before_the_call_blocks_it(tmp_path, monkeypatch):
    t = Transport(message(json.dumps(INVOICE)))
    p = make_provider(tmp_path, t)
    with monkeypatch.context() as patched:
        patched.setattr(spend_module.os, "replace", _disk_full)
        with pytest.raises(LLMProviderError, match="could not write spend ledger"):
            p.complete("extract", "p", "doc-1")
    assert t.requests == []


def test_a_response_whose_cost_cannot_be_read_is_a_provider_error_and_keeps_the_reservation(tmp_path, monkeypatch):
    def unreadable(model, usage):
        raise SpendLedgerError("usage field input_tokens is not a valid token count: '7'")

    monkeypatch.setattr(ap, "cost_usd", unreadable)
    p = make_provider(tmp_path, Transport(message(json.dumps(INVOICE))), max_retries=0)
    with pytest.raises(LLMProviderError, match="token count"):
        p.complete("extract", "p", "doc-1")
    assert ledger_total(tmp_path) == estimate("p")


def test_a_programming_error_is_not_converted_into_a_provider_error(tmp_path, monkeypatch):
    # complete() converts the failures it expects (API, ledger and pricing errors); it does not catch Exception, which
    # would also hide bugs that callers deliberately leave uncaught.
    def buggy(self, reservation, actual):
        raise RuntimeError("a bug")

    monkeypatch.setattr(SpendTracker, "settle", buggy)
    with pytest.raises(RuntimeError, match="a bug"):
        make_provider(tmp_path, Transport(message(json.dumps(INVOICE)))).complete("extract", "p", "doc-1")


def test_extraction_routes_a_ledger_failure_after_a_billed_call_to_a_structured_provider_error(tmp_path, monkeypatch):
    with monkeypatch.context() as patched:
        p = _ledger_write_fails_once_the_request_is_in(tmp_path, monkeypatch, patched)
        with pytest.raises(ExtractionError) as info:
            extract_invoice(_reader_output(), p)
    assert info.value.stage == "provider" and "spend ledger" in info.value.message


def test_gl_coding_falls_back_to_the_keyword_coder_when_the_ledger_fails_after_a_billed_call(tmp_path, monkeypatch):
    with monkeypatch.context() as patched:
        p = _ledger_write_fails_once_the_request_is_in(tmp_path, monkeypatch, patched)
        codes = code_lines(["Cloud hosting", "Foam board posters"], load_chart(), p, "doc-1", vendor_name="Acme")
    assert [c.source for c in codes] == ["keyword_fallback"] * 2
    assert all("spend ledger" in c.reason for c in codes)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_document_intake_sends_a_ledger_failure_to_human_review_with_zero_confidence(tmp_path, monkeypatch):
    state = InvoiceState(invoice_id="x")
    with monkeypatch.context() as patched:
        p = _ledger_write_fails_once_the_request_is_in(tmp_path, monkeypatch, patched)
        summary = fill_state_from_document(state, EN_001_PDF, p)
    assert summary["extraction"] == "error" and summary["error"]["stage"] == "provider"
    assert state.field_confidence.total_amount == 0.0 and state.extracted_fields.total_amount == 0.0


# --- through the extraction and GL layers ------------------------------------------------------------------------------


def _reader_output(text="Acme Marketing Solutions\nInvoice INV-1001\nTotal 108.00"):
    return ReaderOutput(doc_id="doc-1", text=text, method="pdftotext")


def test_extraction_validates_the_models_json_against_the_extraction_schema(tmp_path):
    provider = make_provider(tmp_path, Transport(message(json.dumps(INVOICE))))
    inv = extract_invoice(_reader_output(), provider)
    assert inv.vendor_name == "Acme Marketing Solutions" and inv.total == 108.0 and len(inv.line_items) == 1


def test_a_model_that_cannot_read_the_total_returns_null_and_extraction_fails_rather_than_inventing_one(tmp_path):
    provider = make_provider(tmp_path, Transport(message(json.dumps({**INVOICE, "total": None}))))
    with pytest.raises(ExtractionError) as info:
        extract_invoice(_reader_output(), provider)
    assert info.value.stage == "validation"


def test_a_failed_call_is_an_extraction_error_at_the_provider_stage(tmp_path):
    provider = make_provider(tmp_path, Transport(api_error(401, "authentication_error")))
    with pytest.raises(ExtractionError) as info:
        extract_invoice(_reader_output(), provider)
    assert info.value.stage == "provider"


def test_the_cap_surfaces_as_an_extraction_error_not_an_unhandled_exception(tmp_path):
    provider = make_provider(tmp_path, Transport(message("{}")), cap="0.001")
    with pytest.raises(ExtractionError) as info:
        extract_invoice(_reader_output(), provider)
    assert info.value.stage == "provider" and "cap" in info.value.message


def test_gl_coding_uses_valid_model_codes_and_falls_back_for_the_rest(tmp_path):
    chart = load_chart()
    good = chart[0]["account_number"]
    reply = {
        "lines": [
            {"line": 0, "account": good, "confidence": 0.9, "reason": "hosting"},
            {"line": 1, "account": "9999", "confidence": 0.9, "reason": "off chart"},
        ]
    }
    provider = make_provider(tmp_path, Transport(message(json.dumps(reply))))
    codes = code_lines(["Cloud hosting", "Mystery item"], chart, provider, "doc-1", vendor_name="Acme")
    assert [c.source for c in codes] == ["llm", "keyword_fallback"]
    assert codes[0].account == good


def test_gl_coding_falls_back_to_the_keyword_coder_when_the_call_is_refused(tmp_path):
    t = Transport(message(content=[], stop_reason="refusal", stop_details={"type": "refusal", "category": None, "explanation": ""}))
    codes = code_lines(["Cloud hosting"], load_chart(), make_provider(tmp_path, t), "doc-1", vendor_name="Acme")
    assert [c.source for c in codes] == ["keyword_fallback"]
    assert "declined" in codes[0].reason


# --- configuration: key, model, cap from the environment or the 0600 env file -------------------------------------------


def env_file(tmp_path, body, mode=0o600):
    path = tmp_path / "anthropic.env"
    path.write_text(body, encoding="utf-8")
    os.chmod(path, mode)
    return str(path)


FILE_BODY = f"ANTHROPIC_API_KEY={KEY}\nAP_LLM_PROVIDER=anthropic\nAP_LLM_MODEL=claude-haiku-5-5\nAP_LLM_SPEND_CAP_USD=10\n"


def build(tmp_path, environ=None, body=FILE_BODY, mode=0o600, **kwargs):
    return build_anthropic_provider(
        environ={} if environ is None else environ,
        env_file=env_file(tmp_path, body, mode) if body is not None else str(tmp_path / "absent.env"),
        ledger_path=str(tmp_path / "spend.json"),
        **kwargs,
    )


def test_the_key_model_and_cap_load_from_the_env_file(tmp_path):
    t = Transport(message(json.dumps(INVOICE)))
    p = build(tmp_path, http_client=httpx2.Client(transport=httpx2.MockTransport(t)))
    p.complete("extract", "p", "doc-1")
    assert t.requests[0].headers["x-api-key"] == KEY
    assert t.bodies[0]["model"] == "claude-haiku-5-5"
    assert p.tracker.cap_usd == D("10") and p.tracker.run_cap_usd is None


def test_environment_variables_beat_the_file(tmp_path):
    env = {"ANTHROPIC_API_KEY": "sk-from-env", "AP_LLM_SPEND_CAP_USD": "2.5", "AP_LLM_MODEL": "claude-haiku-5-5"}
    t = Transport(message(json.dumps(INVOICE)))
    p = build(tmp_path, environ=env, http_client=httpx2.Client(transport=httpx2.MockTransport(t)))
    p.complete("extract", "p", "doc-1")
    assert t.requests[0].headers["x-api-key"] == "sk-from-env"
    assert p.tracker.cap_usd == D("2.5")


def test_no_env_file_is_needed_when_the_environment_has_everything(tmp_path):
    env = {"ANTHROPIC_API_KEY": "k", "AP_LLM_SPEND_CAP_USD": "3"}
    p = build(tmp_path, environ=env, body=None)
    assert p.tracker.cap_usd == D("3") and p.model == "claude-haiku-5-5"


def test_a_per_run_cap_is_applied_on_top_of_the_configured_cap(tmp_path):
    p = build(tmp_path, max_usd=0.25)
    assert p.tracker.cap_usd == D("10") and p.tracker.run_cap_usd == D("0.25")


def test_a_missing_key_is_a_config_error_that_names_the_sources_but_no_secret(tmp_path):
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        build(tmp_path, body="AP_LLM_SPEND_CAP_USD=10\n")
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        build(tmp_path, body=None)


def test_a_missing_cap_is_a_config_error_so_a_live_call_can_never_run_uncapped(tmp_path):
    with pytest.raises(ProviderConfigError, match="AP_LLM_SPEND_CAP_USD"):
        build(tmp_path, body=f"ANTHROPIC_API_KEY={KEY}\n")


@pytest.mark.parametrize("cap", ["0", "-5", "ten", "nan"])
def test_a_bad_cap_is_a_config_error(tmp_path, cap):
    with pytest.raises(ProviderConfigError, match="AP_LLM_SPEND_CAP_USD"):
        build(tmp_path, body=f"ANTHROPIC_API_KEY={KEY}\nAP_LLM_SPEND_CAP_USD={cap}\n")


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_a_bad_per_run_cap_is_a_config_error(tmp_path, bad):
    with pytest.raises(ProviderConfigError, match="max-usd"):
        build(tmp_path, max_usd=bad)


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_an_env_file_readable_by_others_is_refused(tmp_path, mode):
    with pytest.raises(ProviderConfigError, match="0600") as info:
        build(tmp_path, mode=mode)
    assert KEY not in str(info.value)


def test_an_owner_only_env_file_with_a_looser_owner_mode_is_accepted(tmp_path):
    build(tmp_path, mode=0o400)


def test_the_secret_never_appears_in_the_providers_repr_or_str(tmp_path):
    p = build(tmp_path)
    assert KEY not in repr(p) and KEY not in str(p)
    assert KEY not in repr(p.tracker)


def test_the_model_effort_and_ceiling_come_from_the_environment(tmp_path):
    env = {"AP_LLM_MODEL": "claude-haiku-5-5", "AP_LLM_EFFORT": "high", "AP_LLM_MAX_TOKENS": "4096"}
    p = build(tmp_path, environ=env)
    assert (p.effort, p.max_tokens) == ("high", 4096)


@pytest.mark.parametrize("name, value", [("AP_LLM_EFFORT", "extreme"), ("AP_LLM_MAX_TOKENS", "0"), ("AP_LLM_MAX_TOKENS", "lots"), ("AP_LLM_MAX_TOKENS", "999999")])
def test_bad_effort_or_ceiling_values_are_config_errors(tmp_path, name, value):
    with pytest.raises(ProviderConfigError, match=name):
        build(tmp_path, environ={name: value})


def test_a_model_the_counter_cannot_price_is_a_config_error(tmp_path):
    with pytest.raises(ProviderConfigError, match="price"):
        build(tmp_path, environ={"AP_LLM_MODEL": "claude-opus-5-5"})


# --- selection through get_provider --------------------------------------------------------------------------------------


def test_get_provider_builds_the_live_provider_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AP_INTAKE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AP_LLM_SPEND_CAP_USD", "10")
    assert isinstance(get_provider(), AnthropicProvider)
    assert isinstance(get_provider(max_usd=0.5), AnthropicProvider)


def test_get_provider_without_a_key_or_cap_is_a_value_error_the_callers_already_treat_as_misconfiguration(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_INTAKE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AP_LLM_SPEND_CAP_USD", raising=False)
    with pytest.raises(ValueError):
        get_provider("anthropic")
    assert issubclass(ProviderConfigError, ValueError)


def test_the_fixture_default_never_reads_the_env_file_or_builds_a_client(tmp_path, monkeypatch):
    monkeypatch.delenv("AP_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AP_INTAKE_ENV_FILE", str(tmp_path / "must-not-be-read.env"))
    (tmp_path / "must-not-be-read.env").write_text(FILE_BODY, encoding="utf-8")
    os.chmod(tmp_path / "must-not-be-read.env", 0o600)
    assert type(get_provider(fixtures_dir=str(tmp_path))).__name__ == "FixtureProvider"

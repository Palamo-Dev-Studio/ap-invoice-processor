# ABOUTME: Tests for the eval's live-model mode, run against the real AnthropicProvider with the HTTP layer mocked.
# ABOUTME: Checks labelling and file separation, integrity rules, the spend cap stopping a run, and that nothing live ever runs.
import functools
import json
import os
import re
import sys
import time

import anthropic
import httpx2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import extraction_eval as ee  # noqa: E402
import extraction_scoring as sc  # noqa: E402
from ap_invoice_processor.llm import anthropic_provider as ap  # noqa: E402
from ap_invoice_processor.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from ap_invoice_processor.llm.provider import FixtureProvider  # noqa: E402
from ap_invoice_processor.llm.spend import SpendTracker  # noqa: E402
from ap_invoice_processor.reader import ReaderOutput  # noqa: E402
from decimal import Decimal  # noqa: E402

FIXTURE_LLM = os.path.join(ROOT, "tests", "fixtures", "llm")
READER_TEXT_DIR = os.path.join(ROOT, "tests", "fixtures", "reader_text")
DOC_PATHS = ee.list_documents()
DOC_IDS = [os.path.splitext(os.path.basename(p))[0] for p in DOC_PATHS]
D = Decimal


def saved_reader(path):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(READER_TEXT_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
        text = f.read()
    method = "tesseract" if re.search(r"-(scan|photo)$", doc_id) else "pdftotext"
    return ReaderOutput(doc_id=doc_id, text=text, method=method, ocr_lang="eng+spa" if method == "tesseract" else None)


SAVED_TEXT = {doc_id: saved_reader(f"{doc_id}.pdf").text for doc_id in DOC_IDS}


def _fixture(task, doc_id):
    with open(os.path.join(FIXTURE_LLM, task, f"{doc_id}.json"), encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


class ReplayingModel:
    """A stand-in for the API: answers each request with the hand-authored fixture for the document it is about.

    Extraction requests are matched to a document by the reader text inside the prompt; the GL request that
    follows belongs to the same document. This drives the whole live path (request building, JSON parsing,
    validation, pricing, ledger) with known answers and no network.
    """

    def __init__(self, usage=None):
        self.requests = []
        self.current_doc = None
        self.usage = usage or {"input_tokens": 1200, "output_tokens": 600}

    def __call__(self, request):
        self.requests.append(request)
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        schema = body["output_config"]["format"]["schema"]
        if schema == ap.EXTRACT_SCHEMA:
            matches = [d for d, text in SAVED_TEXT.items() if text in prompt]
            assert len(matches) == 1, f"prompt matched {matches}"
            self.current_doc = matches[0]
            payload = _fixture("extract", self.current_doc)
        else:
            payload = _fixture("gl", self.current_doc)
        return httpx2.Response(
            200,
            json={
                "id": "msg_replay", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": self.usage,
            },
        )


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def live_provider(tmp_path, model, cap="10", run_cap=None):
    client = anthropic.Anthropic(api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(model)), max_retries=0)
    tracker = SpendTracker(str(tmp_path / "spend.json"), cap_usd=D(cap), run_cap_usd=None if run_cap is None else D(run_cap))
    return AnthropicProvider(client, tracker)


def run_live(tmp_path, **kwargs):
    model = ReplayingModel(kwargs.pop("usage", None))
    provider = live_provider(tmp_path, model, **kwargs)
    return provider, model, ee.run_eval(provider, reader=saved_reader)


# --- labelling and separation from the plumbing check -----------------------------------------------------------------


def test_the_live_banner_carries_every_required_statement(tmp_path):
    provider = live_provider(tmp_path, ReplayingModel())
    banner = ee.banner_for(provider)
    for phrase in (
        "LIVE MODEL RUN", "claude-haiku-5-5", "never sent to it", "not a general accuracy claim",
        "No comparison with the keyword coder", "unscorable", "different report", "do not merge",
    ):
        assert phrase in banner, phrase
    assert banner != ee.BANNER


def test_the_fixture_banner_is_unchanged_in_kind_and_unapproved_providers_are_still_refused(tmp_path):
    assert ee.banner_for(FixtureProvider(FIXTURE_LLM)) == ee.BANNER

    class Unapproved:
        def complete(self, task, prompt, doc_id):
            raise AssertionError

    with pytest.raises(NotImplementedError):
        ee.banner_for(Unapproved())


def test_a_live_report_is_labelled_live_and_never_carries_the_plumbing_wording(tmp_path):
    provider, _, scored = run_live(tmp_path)
    text = ee.render_report(scored)
    assert text.startswith(ee.banner_for(provider))
    assert "LIVE MODEL ACCURACY" in text
    assert ee.BANNER not in text and "no model ran" not in text and "NOT model accuracy" not in text
    assert scored["mode"] == "live"


def test_a_fixture_report_has_no_live_section(tmp_path):
    scored = ee.run_eval(FixtureProvider(FIXTURE_LLM), reader=saved_reader)
    text = ee.render_report(scored)
    assert "LIVE MODEL" not in text and scored["mode"] == "fixture"


def test_live_and_fixture_reports_go_to_different_files_and_neither_overwrites_the_other(tmp_path):
    out = str(tmp_path / "out")
    fixture_scored = ee.run_eval(FixtureProvider(FIXTURE_LLM), reader=saved_reader)
    fixture_paths = ee.write_reports(fixture_scored, out)
    with open(fixture_paths[0], encoding="utf-8") as f:
        fixture_text = f.read()
    _, _, live_scored = run_live(tmp_path)
    live_paths = ee.write_reports(live_scored, out)
    assert set(fixture_paths).isdisjoint(live_paths)
    assert [os.path.basename(p) for p in live_paths] == ["extraction_eval_live.txt", "extraction_eval_live.json"]
    with open(fixture_paths[0], encoding="utf-8") as f:
        assert f.read() == fixture_text
    with open(live_paths[1], encoding="utf-8") as f:
        raw = json.load(f)
    assert list(raw)[0] == "banner" and raw["mode"] == "live" and raw["live"]["model"] == "claude-haiku-5-5"


# --- integrity rules ----------------------------------------------------------------------------------------------------


def test_the_live_path_adds_nothing_the_fixture_path_does_not_when_the_model_answers_identically(tmp_path):
    # Replaying the fixtures through the real provider (schema request, JSON parse, validation) must reproduce the
    # fixture run's scores exactly, so nothing in the live path drops, rewrites or invents a value.
    _, _, live = run_live(tmp_path)
    fixture = ee.run_eval(FixtureProvider(FIXTURE_LLM), reader=saved_reader)
    assert live["summary"] == fixture["summary"]
    assert live["documents"] == fixture["documents"]


class Recording:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, task, prompt, doc_id):
        self.calls.append((task, prompt, doc_id))
        return self.inner.complete(task, prompt, doc_id)


def _scalars(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _scalars(v)
    elif isinstance(value, list):
        for v in value:
            yield from _scalars(v)
    elif value is not None:
        yield str(value)


def test_nothing_the_model_is_sent_carries_ground_truth_or_labels(tmp_path):
    provider = Recording(live_provider(tmp_path, ReplayingModel()))
    labels = sc.load_gl_labels()
    ee.run_pipeline(DOC_PATHS, provider, reader=saved_reader)
    assert {t for t, _, _ in provider.calls} == {"extract", "gl"} and len(provider.calls) == 80
    for task, prompt, doc_id in provider.calls:
        reference = list(_scalars(sc.load_ground_truth(doc_id))) + [label["basis"] for label in labels[doc_id]]
        leaked = sorted({s for s in reference if len(s) >= 4 and s not in SAVED_TEXT[doc_id] and s in prompt})
        assert leaked == [], (task, doc_id, leaked)


def test_every_request_body_holds_only_the_system_prompt_the_prompt_and_the_schema(tmp_path):
    model = ReplayingModel()
    provider = live_provider(tmp_path, model)
    ee.run_pipeline(DOC_PATHS[:3], provider, reader=saved_reader)
    for request in model.requests:
        body = json.loads(request.content)
        assert set(body) == {"model", "max_tokens", "system", "messages", "output_config"}
        assert len(body["messages"]) == 1


def test_the_live_report_shows_gl_as_counts_never_percentages_and_reports_unscorable_lines(tmp_path):
    _, _, scored = run_live(tmp_path)
    text = ee.render_report(scored)
    gl_section = text.split("4. GL CODING")[1].split("5. ")[0]
    assert "%" not in gl_section
    labels = sc.load_gl_labels()
    unscorable = sum(1 for doc_labels in labels.values() for label in doc_labels if label["account"] is None)
    assert unscorable > 0
    assert f"Unscorable (no fitting account)  {unscorable}" in re.sub(r" {2,}", "  ", gl_section)
    assert scored["summary"]["overall"]["gl"]["unscorable"] == unscorable


def test_the_report_makes_no_llm_versus_keyword_comparison(tmp_path):
    _, _, scored = run_live(tmp_path)
    text = ee.render_report(scored).lower()
    assert "keyword" in text  # only to disclaim it and to count fallbacks
    for phrase in ("better than", "outperform", "versus keyword", "vs keyword", "vs. keyword", "keyword coder scored", "keyword accuracy"):
        assert phrase not in text
    assert "keyword" not in json.dumps(scored["summary"]).lower()


# --- spend and completeness ---------------------------------------------------------------------------------------------


def test_the_report_states_the_run_spend_and_the_ledger_total(tmp_path):
    provider, model, scored = run_live(tmp_path)
    per_call = (D(1200) * D("0.10") + D(600) * D("0.50")) / D(1_000_000)
    assert provider.tracker.run_spent_usd == per_call * 80 == len(model.requests) * per_call
    text = ee.render_report(scored)
    assert "5. LIVE RUN COST" in text
    assert scored["live"]["run_spent_usd"] == format(per_call * 80, "f")
    assert scored["live"]["cap_usd"] == "10" and scored["live"]["calls"] == 80
    assert "$0.0" in text


def test_a_run_the_cap_stops_is_reported_incomplete_with_the_unrun_documents_and_no_partial_document(tmp_path):
    provider, model, scored = run_live(tmp_path, run_cap="0.03")
    ran = scored["summary"]["overall"]["documents"]
    not_run = scored["not_run"]
    assert 0 < ran < 40 and ran + len(not_run) == 40
    assert set(not_run).isdisjoint(d["doc_id"] for d in scored["documents"])
    assert not_run == [i for i in DOC_IDS if i not in {d["doc_id"] for d in scored["documents"]}]
    assert provider.cap_reached is True
    # a document the cap interrupted is dropped whole, so no scored document carries a cap-induced failure or fallback
    assert scored["summary"]["overall"]["extraction_errors"] == 0
    assert scored["summary"]["overall"]["gl"]["fallback"] == 0
    text = ee.render_report(scored)
    assert "INCOMPLETE RUN" in text and not_run[0] in text
    assert provider.tracker.run_spent_usd <= D("0.03")


def test_a_cap_that_is_already_spent_runs_nothing_and_sends_no_request(tmp_path):
    SpendTracker(str(tmp_path / "spend.json"), cap_usd=D("10")).record(D("9.999"))
    provider, model, scored = run_live(tmp_path, cap="10")
    assert model.requests == []
    assert scored["summary"]["overall"]["documents"] == 0 and scored["not_run"] == DOC_IDS


def test_provider_failures_are_counted_by_stage_and_the_failed_documents_stay_in_every_denominator(tmp_path):
    def always_unauthorized(request):
        return httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})

    provider = live_provider(tmp_path, always_unauthorized)
    scored = ee.run_eval(provider, reader=saved_reader)
    overall = scored["summary"]["overall"]
    assert overall["extraction_errors"] == 40 and scored["not_run"] == []
    assert overall[sc.ALL_HEADER]["n"] == 329 and overall[sc.ALL_HEADER]["strict"] == 0
    text = ee.render_report(scored)
    assert "provider: 40" in text and "401" in text


# --- the command line ---------------------------------------------------------------------------------------------------


def test_a_live_run_needs_a_per_run_cap(monkeypatch, tmp_path, capsys):
    model = ReplayingModel()
    monkeypatch.setattr(ee, "get_provider", lambda *a, **k: live_provider(tmp_path, model))
    assert ee.main(["--provider", "anthropic", "--out", str(tmp_path / "out")]) == 2
    assert "--max-usd" in capsys.readouterr().err
    assert model.requests == []


def test_max_usd_is_refused_for_the_fixture_provider(tmp_path, capsys):
    assert ee.main(["--provider", "fixture", "--max-usd", "1", "--out", str(tmp_path)]) == 2
    assert "live" in capsys.readouterr().err


def test_a_live_provider_that_cannot_be_configured_is_a_refusal_not_a_crash(tmp_path, capsys):
    assert ee.main(["--provider", "anthropic", "--max-usd", "1", "--out", str(tmp_path)]) == 2
    assert "not run" in capsys.readouterr().err


def test_main_runs_the_live_mode_end_to_end_with_a_mocked_model(monkeypatch, tmp_path, capsys):
    model = ReplayingModel()
    seen = {}

    def fake_get_provider(name=None, fixtures_dir=None, max_usd=None):
        seen["args"] = (name, max_usd)
        return live_provider(tmp_path, model, run_cap=str(max_usd))

    monkeypatch.setattr(ee, "get_provider", fake_get_provider)
    monkeypatch.setattr(ee, "run_eval", functools.partial(ee.run_eval, reader=saved_reader))
    out = tmp_path / "out"
    assert ee.main(["--provider", "anthropic", "--max-usd", "0.5", "--out", str(out)]) == 0
    assert seen["args"] == ("anthropic", 0.5)
    stdout = capsys.readouterr().out
    assert stdout.startswith("LIVE MODEL RUN") and "LIVE MODEL ACCURACY" in stdout
    assert "Spend this run" in stdout
    assert sorted(os.listdir(out)) == ["extraction_eval_live.json", "extraction_eval_live.txt"]


def test_main_exits_3_when_the_cap_stops_the_run(monkeypatch, tmp_path, capsys):
    model = ReplayingModel()
    monkeypatch.setattr(ee, "get_provider", lambda name=None, fixtures_dir=None, max_usd=None: live_provider(tmp_path, model, run_cap="0.03"))
    monkeypatch.setattr(ee, "run_eval", functools.partial(ee.run_eval, reader=saved_reader))
    assert ee.main(["--provider", "anthropic", "--max-usd", "0.03", "--out", str(tmp_path / "out")]) == 3
    assert "INCOMPLETE RUN" in capsys.readouterr().out

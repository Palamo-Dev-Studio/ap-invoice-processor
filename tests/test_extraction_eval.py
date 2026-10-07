# ABOUTME: Tests for the offline extraction eval: scorer behaviour, disclosure banner, reader-only isolation, negative controls.
# ABOUTME: The controls prove the scorer separates good from blank/wrong fixtures; they say nothing about any model.
import builtins
import copy
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import extraction_eval as ee  # noqa: E402
import extraction_scoring as sc  # noqa: E402
from ap_invoice_processor.llm.coercion import parse_amount  # noqa: E402
from ap_invoice_processor.llm.extraction import ExtractedInvoice, ExtractedLineItem  # noqa: E402
from ap_invoice_processor.llm.gl import GLCode, load_chart  # noqa: E402
from ap_invoice_processor.llm.provider import AnthropicProvider, FixtureProvider  # noqa: E402
from ap_invoice_processor.reader import ReaderOutput  # noqa: E402

FIXTURE_LLM = os.path.join(ROOT, "tests", "fixtures", "llm")
READER_TEXT_DIR = os.path.join(ROOT, "tests", "fixtures", "reader_text")
CORPUS = sc.CORPUS_DIR
DOC_PATHS = ee.list_documents()
DOC_IDS = [os.path.splitext(os.path.basename(p))[0] for p in DOC_PATHS]
CHART_NUMBERS = [a["account_number"] for a in load_chart()]

# Thresholds for the negative controls. A control that is wrong or empty everywhere must score at or near zero.
CONTROL_CEILING = 0.05


def _fixture_reader(path):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(READER_TEXT_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
        text = f.read()
    method = "tesseract" if re.search(r"-(scan|photo)$", doc_id) else "pdftotext"
    return ReaderOutput(doc_id=doc_id, text=text, method=method, ocr_lang="eng" if method == "tesseract" else None)


def _eval(fixtures_dir):
    return ee.run_eval(FixtureProvider(fixtures_dir), reader=_fixture_reader)


def _gt(doc_id):
    return sc.load_ground_truth(doc_id)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- control fixture builders (generated per test in a tmp dir; never committed) ---------------------------------


def _fresh_copy(tmp_path):
    dest = str(tmp_path / "llm")
    shutil.copytree(FIXTURE_LLM, dest)
    return dest


def _blank_extract(null_total):
    # total is a required field: None makes every document fail validation, 0 yields a valid but empty extraction.
    blank = {k: None for k in sc.HEADER_FIELDS}
    blank["total"] = None if null_total else 0
    blank["line_items"] = []
    return blank


def _perturb_extract(fx):
    out = copy.deepcopy(fx)
    for key in ("vendor_name", "invoice_number", "po_number"):
        if out.get(key) is not None:
            out[key] = "ZZ-" + str(out[key])[::-1]
    for key in ("invoice_date", "due_date"):
        if out.get(key) is not None:
            out[key] = "2031-01-01"
    out["currency"] = "JPY"
    for key in ("subtotal", "tax", "total"):
        if out.get(key) is not None:
            out[key] = parse_amount(out[key]) + 0.37
    for i, item in enumerate(out["line_items"]):
        item["description"] = f"Wrong line {i}"
        item["quantity"] = parse_amount(item["quantity"]) + 7
        item["unit_price"] = parse_amount(item["unit_price"]) + 0.37
        item["amount"] = parse_amount(item["amount"]) + 0.37
    return out


def _gl_entries(accounts):
    return {"lines": [{"line": i, "account": a, "confidence": 0.9, "reason": "control"} for i, a in enumerate(accounts)]}


def _wrong_account(expected):
    return next(n for n in CHART_NUMBERS if n != expected)


def _build_blank(tmp_path, null_total):
    d = _fresh_copy(tmp_path)
    for doc_id in DOC_IDS:
        _write(os.path.join(d, "extract", f"{doc_id}.json"), _blank_extract(null_total))
        _write(os.path.join(d, "gl", f"{doc_id}.json"), {"lines": []})
    return d


def _build_perturbed(tmp_path):
    d = _fresh_copy(tmp_path)
    labels = sc.load_gl_labels()
    for doc_id in DOC_IDS:
        extract = _perturb_extract(_read(os.path.join(FIXTURE_LLM, "extract", f"{doc_id}.json")))
        _write(os.path.join(d, "extract", f"{doc_id}.json"), extract)
        _write(
            os.path.join(d, "gl", f"{doc_id}.json"),
            _gl_entries([_wrong_account(label["account"]) for label in labels[doc_id]]),
        )
    return d


def _base_id(doc_id):
    return re.sub(r"-(scan|photo)$", "", doc_id)


def _build_shifted(tmp_path):
    """Each document gets the fixtures of the next document that is not a variant of the same invoice."""
    d = _fresh_copy(tmp_path)
    for k, doc_id in enumerate(DOC_IDS):
        step = 1
        while _base_id(DOC_IDS[(k + step) % len(DOC_IDS)]) == _base_id(doc_id):
            step += 1
        donor = DOC_IDS[(k + step) % len(DOC_IDS)]
        for task in ("extract", "gl"):
            _write(os.path.join(d, task, f"{doc_id}.json"), _read(os.path.join(FIXTURE_LLM, task, f"{donor}.json")))
    return d


def _build_oracle(tmp_path):
    """Fixtures copied from ground truth and labels: the top of the scale, for the scorer's upper bound only."""
    d = _fresh_copy(tmp_path)
    labels = sc.load_gl_labels()
    for doc_id in DOC_IDS:
        gt = _gt(doc_id)
        extract = {k: gt[k] for k in sc.HEADER_FIELDS}
        extract["line_items"] = [{k: li[k] for k in sc.LINE_FIELDS} for li in gt["line_items"]]
        _write(os.path.join(d, "extract", f"{doc_id}.json"), extract)
        _write(os.path.join(d, "gl", f"{doc_id}.json"), _gl_entries([l["account"] or "6100" for l in labels[doc_id]]))
    return d


@pytest.fixture(scope="module")
def real():
    return _eval(FIXTURE_LLM)


def _rates(scored):
    s = scored["summary"]["overall"]
    return s["extraction_strict_rate"], s["extraction_lenient_rate"], s["gl_match_rate"]


# --- scorer unit tests ------------------------------------------------------------------------------------------


def test_text_normalisation_is_case_and_whitespace_insensitive_but_keeps_accents():
    assert sc.strict_text("  Papeleria   EL Roble ") == sc.strict_text("papeleria el roble")
    assert sc.strict_text("Papelería") != sc.strict_text("Papeleria")
    assert sc.lenient_text("Papelería") == sc.lenient_text("papeleria")


def test_strict_and_lenient_columns_are_reported_separately():
    assert sc.fields_match("vendor_name", "Papeleria El Roble", "Papelería El Roble") == (False, True)
    assert sc.fields_match("vendor_name", "papelería  el roble", "Papelería El Roble") == (True, True)
    assert sc.fields_match("vendor_name", None, "X") == (False, False)
    # Line descriptions follow the same rule: strict keeps accents, only the lenient column folds them.
    assert sc.fields_match("description", "Papeleria", "Papelería") == (False, True)


def test_amounts_compare_to_the_cent_not_as_floats():
    assert sc.fields_match("total", 20359.43, 20359.43)[0]
    assert sc.fields_match("total", 0.1 + 0.2, 0.3)[0]
    assert not sc.fields_match("total", 20359.44, 20359.43)[0]
    assert sc.fields_match("tax", 0, 0.0)[0]


def test_dates_and_currency_compare_exactly_after_normalisation():
    assert sc.fields_match("invoice_date", "2026-05-17", "2026-05-17")[0]
    assert not sc.fields_match("invoice_date", "2026-05-18", "2026-05-17")[0]
    assert sc.fields_match("currency", "usd", "USD")[0]
    assert not sc.fields_match("currency", "MXN", "USD")[0]


def _line(description, quantity=1, unit_price=1.0, amount=1.0):
    return ExtractedLineItem(description=description, quantity=quantity, unit_price=unit_price, amount=amount)


def test_line_matching_prefers_description_then_position():
    truth = [{"description": "Alpha"}, {"description": "Beta"}, {"description": "Gamma"}]
    assert sc.match_lines([_line("Alpha"), _line("Beta"), _line("Gamma")], truth) == {0: 0, 1: 1, 2: 2}
    assert sc.match_lines([_line("Gamma"), _line("Alpha")], truth) == {0: 2, 1: 0}
    # A damaged description still pairs by position, so its cells are scored (as misses) rather than dropped.
    assert sc.match_lines([_line("Alpha"), _line("B3ta"), _line("Gamma")], truth) == {0: 0, 1: 1, 2: 2}
    assert sc.match_lines([_line("Alpha"), _line("Beta"), _line("Gamma"), _line("Extra")], truth) == {0: 0, 1: 1, 2: 2}


def _result(doc_id="d1", **overrides):
    ext = ExtractedInvoice(
        vendor_name="Acme", invoice_number="A-1", invoice_date="2026-01-02", currency="USD", po_number=None,
        subtotal=10, tax=0, total=10, line_items=[_line("Widget", 1, 10, 10)],
    )
    code = GLCode(account="6100", account_name="x", confidence=0.9, reason="r", source="llm")
    base = ee.DocResult(doc_id, extraction=ext, gl_codes=[code])
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


_GT = {
    "language": "en", "variant": "pdf", "vendor_name": "Acme", "invoice_number": "A-1", "invoice_date": "2026-01-02",
    "due_date": None, "currency": "USD", "po_number": None, "subtotal": 10, "tax": 0, "total": 10,
    "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 10, "amount": 10}],
}


def test_null_truth_cells_are_not_scorable_and_a_returned_value_is_flagged_spurious():
    res = _result()
    res.extraction = res.extraction.model_copy(update={"po_number": "PO-9"})
    doc = sc.score_document(res, _GT, [{"line": 0, "account": "6100"}])
    assert doc["header"]["po_number"] == {"scorable": False, "spurious": True, "got": "PO-9", "expected": None}
    assert doc["header"]["due_date"]["scorable"] is False and doc["header"]["due_date"]["spurious"] is False
    agg = sc.aggregate([doc])["overall"]
    assert agg["header"]["po_number"]["n"] == 0 and agg["header"]["po_number"]["spurious"] == 1
    assert agg["header"]["vendor_name"]["strict"] == 1


@pytest.mark.parametrize(
    "code,expected_status",
    [
        (GLCode(account="6100", account_name="x", confidence=0.9, reason="r", source="llm"), "correct"),
        (GLCode(account="6500", account_name="x", confidence=0.9, reason="r", source="llm"), "incorrect"),
        (GLCode(account="6100", account_name="x", confidence=0.3, reason="r", source="keyword_fallback"), "fallback"),
    ],
)
def test_gl_status_per_source_and_account(code, expected_status):
    doc = sc.score_document(_result(gl_codes=[code]), _GT, [{"line": 0, "account": "6100"}])
    assert doc["gl"][0]["status"] == expected_status


def test_gl_null_label_is_unscorable_and_missing_line_is_counted():
    doc = sc.score_document(_result(), _GT, [{"line": 0, "account": None}])
    assert doc["gl"][0]["status"] == "unscorable"
    assert sc.aggregate([doc])["overall"]["gl"]["scorable"] == 0
    empty = sc.score_document(_result(extraction=None, gl_codes=[], error={"stage": "provider"}), _GT, [{"line": 0, "account": "6100"}])
    assert empty["gl"][0]["status"] == "no_line"
    assert empty["extraction_error"] == {"stage": "provider"}
    assert sc.aggregate([empty])["overall"]["gl"]["no_line"] == 1


def test_gl_codes_are_scored_against_their_own_paired_line_and_extra_lines_are_counted():
    # Extracted order is reversed relative to ground truth and every line has a different account, so a lookup by
    # ground-truth index instead of by the paired extracted index scores the wrong code against each label.
    truth = {
        **_GT,
        "line_items": [
            {"description": "Alpha", "quantity": 1, "unit_price": 1, "amount": 1},
            {"description": "Beta", "quantity": 1, "unit_price": 1, "amount": 1},
            {"description": "Gamma", "quantity": 1, "unit_price": 1, "amount": 1},
        ],
    }
    extracted = [_line("Gamma"), _line("Beta"), _line("Alpha"), _line("Extra")]
    codes = [GLCode(account=a, account_name="x", confidence=0.9, reason="r", source="llm") for a in ("6300", "6200", "6100", "6400")]
    res = _result(gl_codes=codes)
    res.extraction = res.extraction.model_copy(update={"line_items": extracted})
    labels = [{"line": 0, "account": "6100"}, {"line": 1, "account": "6200"}, {"line": 2, "account": "6300"}]
    doc = sc.score_document(res, truth, labels)
    assert [(r["line"], r["status"]) for r in doc["gl"]] == [(0, "correct"), (1, "correct"), (2, "correct")]
    assert [line["extracted_line"] for line in doc["lines"]] == [2, 1, 0]
    assert doc["extra_extracted_lines"] == 1
    assert sc.aggregate([doc])["overall"]["line_counts"] == {"truth": 3, "extra": 1}


# --- end to end over the corpus with the real fixtures -----------------------------------------------------------


def test_real_fixture_run_covers_the_corpus_and_reports_every_row(real):
    summary = real["summary"]
    assert len(real["documents"]) == 40
    assert {k: summary[k]["documents"] for k in ("lang:en", "lang:es", "variant:pdf", "variant:scan", "variant:photo")} == {
        "lang:en": 24, "lang:es": 16, "variant:pdf": 30, "variant:scan": 6, "variant:photo": 4,
    }
    overall = summary["overall"]
    assert overall["extraction_errors"] == 0
    assert overall["gl"]["unscorable"] == 42 and overall["gl"]["scorable"] == 56
    assert overall["gl"]["fallback"] == 0
    assert overall[sc.ALL_HEADER]["lenient"] >= overall[sc.ALL_HEADER]["strict"]
    # Accent folding recovers cells the strict column misses, and that shows up only in the lenient column.
    assert overall["header"]["vendor_name"]["lenient"] > overall["header"]["vendor_name"]["strict"]


def test_report_has_the_documented_sections_and_no_keyword_score(real):
    text = ee.render_report(real)
    for heading in ("1. HEADER FIELDS", "2. LINE ITEMS", "3. BY LANGUAGE AND BY VARIANT", "4. GL CODING",
                    "Unscorable (no fitting account)", "keyword-coder fallback lines:", "lenient (accent-folded)"):
        assert heading in text
    for group in ("en", "es", "pdf", "scan", "photo"):
        assert re.search(rf"^{group}\s+\d+", text, re.MULTILINE), group
    assert "44%" not in text and "82%" not in text
    lowered = text.lower()
    assert "delta" not in lowered and "vs keyword" not in lowered and "keyword coder score" not in lowered


def test_nothing_in_the_eval_scores_the_keyword_coder():
    for name in ("extraction_eval.py", "extraction_scoring.py"):
        with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as f:
            source = f.read()
        assert "keyword_code" not in source and "KEYWORD_CONFIDENCE" not in source, name


def test_fallback_lines_are_listed_and_not_scored(tmp_path):
    d = _fresh_copy(tmp_path)
    doc_id = "en-001"
    bad = _read(os.path.join(FIXTURE_LLM, "gl", f"{doc_id}.json"))
    bad["lines"][0]["account"] = "9999"  # off-chart: the GL coder falls back to the keyword coder for this line
    _write(os.path.join(d, "gl", f"{doc_id}.json"), bad)
    scored = _eval(d)
    gl = scored["summary"]["overall"]["gl"]
    assert gl["fallback"] >= 1
    assert f"{doc_id}#0" in ee.render_report(scored)


def test_extraction_failure_is_scored_as_misses_and_listed(tmp_path):
    d = _fresh_copy(tmp_path)
    os.remove(os.path.join(d, "extract", "en-001.json"))
    scored = _eval(d)
    overall = scored["summary"]["overall"]
    assert overall["extraction_errors"] == 1
    # The failed document stays in every denominator: its cells count as misses, they are not dropped.
    assert overall[sc.ALL_HEADER]["n"] == 329 and overall[sc.ALL_LINE]["n"] == 392
    assert "Failed documents" in ee.render_report(scored) and "en-001" in ee.render_report(scored)


# --- disclosure ---------------------------------------------------------------------------------------------------


def test_banner_wording_carries_every_required_caveat():
    for phrase in (
        "PLUMBING CHECK", "offline, fixture-backed", "hand-authored from reader text", "no model ran",
        "NOT model accuracy", "GL fixture/label agreement is not independent (same author)",
        "Spanish scans/photos OCR'd with -l eng",
    ):
        assert phrase in ee.BANNER


def test_reports_start_with_the_banner(real, tmp_path):
    assert ee.render_report(real).startswith(ee.BANNER)
    txt, js = ee.write_reports(real, str(tmp_path / "out"))
    with open(txt, encoding="utf-8") as f:
        assert f.read().startswith(ee.BANNER)
    with open(js, encoding="utf-8") as f:
        raw = f.read()
    assert list(json.loads(raw))[0] == "banner"
    assert json.loads(raw)["banner"] == ee.BANNER


def test_a_provider_without_an_approved_banner_is_refused(capsys):
    with pytest.raises(NotImplementedError):
        ee.banner_for(AnthropicProvider())
    assert ee.main(["--provider", "anthropic"]) == 2
    captured = capsys.readouterr()
    assert "not run" in captured.err and ee.BANNER not in captured.out
    assert ee.main(["--provider", "nope"]) == 2


# --- 5b: reader-only isolation ----------------------------------------------------------------------------------


class RecordingProvider:
    """Wraps the fixture provider and records every (task, prompt, doc_id) it is handed."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

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


def leaked_values(prompt, reader_text, reference):
    """Reference scalars (4+ characters) that are absent from the reader text yet present in the prompt."""
    return sorted({s for s in _scalars(reference) if len(s) >= 4 and s not in reader_text and s in prompt})


def _clean_text(doc_id):
    return _fixture_reader(f"{doc_id}.pdf").text


def _record(reader=_fixture_reader, paths=DOC_PATHS):
    rec = RecordingProvider(FixtureProvider(FIXTURE_LLM))
    results = ee.run_pipeline(paths, rec, reader=reader)
    return rec, results


def test_both_tasks_run_for_every_document_and_prompts_carry_no_ground_truth_only_value():
    rec, results = _record()
    labels = sc.load_gl_labels()
    assert len(results) == 40 and not any(r.error for r in results)
    tasks = {(t, d) for t, _, d in rec.calls}
    assert tasks == {(t, d) for t in ("extract", "gl") for d in DOC_IDS}
    for task, prompt, doc_id in rec.calls:
        gt = _gt(doc_id)
        assert leaked_values(prompt, _clean_text(doc_id), gt) == [], (task, doc_id)
        # GL labels and their stated basis never reach a prompt either.
        basis = [label["basis"] for label in labels[doc_id]]
        assert leaked_values(prompt, _clean_text(doc_id), basis) == [], (task, doc_id)


def test_gl_prompt_holds_only_chart_and_extracted_line_descriptions():
    rec, results = _record()
    by_doc = {r.doc_id: r for r in results}
    for task, prompt, doc_id in rec.calls:
        if task != "gl":
            continue
        for item in by_doc[doc_id].extraction.line_items:
            assert item.description in prompt
        for value in (by_doc[doc_id].extraction.invoice_number, by_doc[doc_id].extraction.vendor_name):
            assert value not in prompt, (doc_id, value)


def test_the_pipeline_never_opens_ground_truth_or_labels(monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def forbidden(*_a, **_k):
        raise AssertionError("the pipeline must not load ground truth or labels")

    monkeypatch.setattr(sc, "load_ground_truth", forbidden)
    monkeypatch.setattr(sc, "load_gl_labels", forbidden)
    monkeypatch.setattr(builtins, "open", spy)
    ee.run_pipeline(DOC_PATHS, FixtureProvider(FIXTURE_LLM), reader=_fixture_reader)
    monkeypatch.undo()
    assert opened, "the spy must have seen the fixture reads"
    assert not [p for p in opened if "ground_truth" in p or "gl_labels" in p], opened


def test_only_the_scorer_module_names_ground_truth_or_labels():
    with open(os.path.join(EVAL_DIR, "extraction_eval.py"), encoding="utf-8") as f:
        pipeline_source = f.read()
    for token in ("ground_truth", "gl_labels", "simulated_extraction"):
        assert token not in pipeline_source, token
    with open(os.path.join(EVAL_DIR, "extraction_scoring.py"), encoding="utf-8") as f:
        scorer_source = f.read()
    assert "ground_truth" in scorer_source and "gl_labels.json" in scorer_source


def test_isolation_check_has_teeth_a_leaky_reader_is_caught():
    leaky_docs = DOC_PATHS[:5]

    def leaky_reader(path):
        ro = _fixture_reader(path)
        return ReaderOutput(doc_id=ro.doc_id, text=ro.text + "\n" + json.dumps(_gt(ro.doc_id)), method=ro.method,
                            ocr_lang=ro.ocr_lang)

    rec, _ = _record(reader=leaky_reader, paths=leaky_docs)
    extract_calls = [(p, d) for t, p, d in rec.calls if t == "extract"]
    assert extract_calls
    for prompt, doc_id in extract_calls:
        assert leaked_values(prompt, _clean_text(doc_id), _gt(doc_id)), doc_id


# --- negative controls ------------------------------------------------------------------------------------------


def test_blank_fixtures_with_null_total_fail_every_document_and_score_zero(tmp_path):
    scored = _eval(_build_blank(tmp_path, null_total=True))
    overall = scored["summary"]["overall"]
    assert overall["extraction_errors"] == 40
    strict, lenient, gl = _rates(scored)
    assert strict <= CONTROL_CEILING and lenient <= CONTROL_CEILING and gl <= CONTROL_CEILING
    # Every document failed, yet every scorable cell and label stays in its denominator, so the rates are real zeros.
    assert overall[sc.ALL_HEADER]["n"] == 329 and overall[sc.ALL_LINE]["n"] == 392
    assert overall["gl"]["no_line"] == 56 and overall["gl"]["scorable"] == 56
    assert gl == 0.0


def test_blank_fixtures_with_empty_values_score_at_or_near_zero(tmp_path):
    scored = _eval(_build_blank(tmp_path, null_total=False))
    assert scored["summary"]["overall"]["extraction_errors"] == 0
    strict, lenient, gl = _rates(scored)
    assert strict <= CONTROL_CEILING and lenient <= CONTROL_CEILING and gl <= CONTROL_CEILING


def test_blank_gl_fixtures_fall_back_and_do_not_score_as_matches(tmp_path):
    d = _fresh_copy(tmp_path)
    for doc_id in DOC_IDS:
        _write(os.path.join(d, "gl", f"{doc_id}.json"), {"lines": []})
    gl = _eval(d)["summary"]["overall"]["gl"]
    assert gl["correct"] == 0 and gl["fallback"] == 56


def test_perturbed_fixtures_score_at_or_near_zero(tmp_path, real):
    scored = _eval(_build_perturbed(tmp_path))
    assert scored["summary"]["overall"]["extraction_errors"] == 0
    strict, lenient, gl = _rates(scored)
    assert strict <= CONTROL_CEILING and lenient <= CONTROL_CEILING and gl <= CONTROL_CEILING
    real_strict, _, real_gl = _rates(real)
    assert real_strict > strict and real_gl > gl


def test_shifted_fixtures_score_well_below_the_real_ones(tmp_path, real):
    scored = _eval(_build_shifted(tmp_path))
    strict, _, gl = _rates(scored)
    real_strict, _, real_gl = _rates(real)
    # The donor is never a variant of the recipient's own invoice, so what survives is coincidence only.
    assert strict < real_strict / 4
    assert gl < real_gl / 2


def test_real_fixtures_score_strictly_above_every_control(tmp_path, real):
    real_strict, real_lenient, real_gl = _rates(real)
    for scored in (
        _eval(_build_blank(tmp_path / "a", True)),
        _eval(_build_blank(tmp_path / "b", False)),
        _eval(_build_perturbed(tmp_path / "c")),
        _eval(_build_shifted(tmp_path / "d")),
    ):
        strict, lenient, gl = _rates(scored)
        assert real_strict > strict and real_lenient > lenient and real_gl > gl


def test_oracle_fixtures_reach_the_top_of_the_scale(tmp_path, real):
    strict, lenient, gl = _rates(_eval(_build_oracle(tmp_path)))
    assert strict == 1.0 and lenient == 1.0 and gl == 1.0
    assert _rates(real)[0] < 1.0


# --- command line, end to end with the real reader ---------------------------------------------------------------


@pytest.mark.skipif(not (shutil.which("pdftotext") and shutil.which("tesseract")), reason="reader binaries not installed")
def test_command_line_run_reads_real_documents_and_writes_reports(tmp_path):
    env = {**os.environ, "PYTHONPATH": ROOT}
    proc = subprocess.run(
        [sys.executable, os.path.join(EVAL_DIR, "extraction_eval.py"), "--provider", "fixture", "--out", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(ee.BANNER)
    assert "44%" not in proc.stdout and "82%" not in proc.stdout
    assert proc.stderr == ""
    with open(tmp_path / ee.REPORT_TXT, encoding="utf-8") as f:
        assert f.read().startswith(ee.BANNER)
    assert list(_read(str(tmp_path / ee.REPORT_JSON)))[0] == "banner"
    # The live reader and the saved reader text agree, so this run matches the fixture-reader run.
    assert _read(str(tmp_path / ee.REPORT_JSON))["summary"]["overall"]["gl"]["scorable"] == 56


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_eval_output_directory_is_gitignored():
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "eval/out/extraction_eval.txt"], cwd=ROOT, timeout=30, check=False,
    )
    assert proc.returncode == 0

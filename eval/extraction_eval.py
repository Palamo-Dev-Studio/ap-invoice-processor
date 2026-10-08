# ABOUTME: End-to-end eval: reader -> LLM extraction -> GL coding, then scored by extraction_scoring.
# ABOUTME: Fixture runs are a plumbing check only; --provider anthropic is a separate, labelled live-model run. Every report opens with a banner.
import argparse
import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Make `python eval/extraction_eval.py` find the sibling scorer and the package without relying on PYTHONPATH.
for _path in (_HERE, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import extraction_scoring as scoring  # noqa: E402
from ap_invoice_processor.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from ap_invoice_processor.llm.extraction import ExtractedInvoice, ExtractionError, extract_invoice  # noqa: E402
from ap_invoice_processor.llm.gl import GLCode, code_lines, load_chart  # noqa: E402
from ap_invoice_processor.llm.provider import FixtureProvider, LLMProvider, get_provider  # noqa: E402
from ap_invoice_processor.reader import ReaderError, ReaderOutput, read_document, resolve_ocr_langs  # noqa: E402

CORPUS_DIR = scoring.CORPUS_DIR
DEFAULT_OUT_DIR = os.path.join(_HERE, "out")
REPORT_TXT = "extraction_eval.txt"
REPORT_JSON = "extraction_eval.json"
# A live run writes to its own files so it can never overwrite, or be mistaken for, the fixture plumbing report.
LIVE_REPORT_TXT = "extraction_eval_live.txt"
LIVE_REPORT_JSON = "extraction_eval_live.json"
EXIT_INCOMPLETE = 3

BANNER = (
    "PLUMBING CHECK — offline, fixture-backed. Fixtures are hand-authored from reader text; no model ran. "
    "These numbers validate the scoring pipeline and wiring only and are NOT model accuracy. "
    "GL fixture/label agreement is not independent (same author). "
    "Fixtures were authored from reader text of Spanish scans/photos OCR'd with -l eng; the reader now defaults to eng+spa."
)


@dataclass
class DocResult:
    """What the pipeline produced for one document. It holds no reference data of any kind."""

    doc_id: str
    extraction: Optional[ExtractedInvoice] = None
    gl_codes: List[GLCode] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None


def is_live(provider: LLMProvider) -> bool:
    return isinstance(provider, AnthropicProvider)


def live_banner(provider: AnthropicProvider) -> str:
    """The disclosure banner for a live-model run. It states what was run and the limits of what it shows."""
    return (
        f"LIVE MODEL RUN — {provider.model} (effort {provider.effort}). A live model read the synthetic corpus "
        "documents and its output is scored against the corpus reference answers, which were never sent to it. "
        "This is a different report from the offline fixture plumbing check (extraction_eval.txt): do not merge, "
        "average or compare the two as one series. One run, one model, synthetic invoices: not a general accuracy "
        "claim. No comparison with the keyword coder is made or implied. GL results are counts against one reader's "
        "labels on a five-account chart; lines with no fitting account are unscorable and reported as such. "
        f"Reader OCR languages: {resolve_ocr_langs()}."
    )


def banner_for(provider: LLMProvider) -> str:
    """The disclosure banner for a run. Only the fixture provider and the live Anthropic provider have approved wording."""
    if isinstance(provider, FixtureProvider):
        return BANNER
    if is_live(provider):
        return live_banner(provider)
    raise NotImplementedError(
        "no disclosure banner is defined for this provider; its wording must be agreed before a run is reported"
    )


def list_documents(corpus_dir: str = CORPUS_DIR) -> List[str]:
    """Every corpus document (PDFs, then image variants), sorted by document id."""
    paths = glob.glob(os.path.join(corpus_dir, "pdf", "*.pdf")) + glob.glob(os.path.join(corpus_dir, "images", "*.png"))
    return sorted(paths, key=lambda p: os.path.splitext(os.path.basename(p))[0])


def run_pipeline(
    doc_paths: Sequence[str],
    provider: LLMProvider,
    reader: Callable[[str], ReaderOutput] = read_document,
    chart: Optional[Sequence[Dict[str, Any]]] = None,
    not_run: Optional[List[str]] = None,
) -> List[DocResult]:
    """Read, extract and GL-code each document. Extraction receives only the ReaderOutput; the GL coder receives
    only the extracted lines, the chart and the extracted vendor name.

    A provider whose spend cap has been reached (`cap_reached`) ends the run: the document it interrupted is
    dropped whole, since its failures and keyword fallbacks would be artefacts of the cap, and that document and
    every later one are appended to `not_run` when a list is given.
    """
    chart = chart if chart is not None else load_chart()
    results: List[DocResult] = []
    doc_ids = [os.path.splitext(os.path.basename(p))[0] for p in doc_paths]
    for position, path in enumerate(doc_paths):
        result = None if _cap_reached(provider) else _run_document(path, doc_ids[position], provider, reader, chart)
        if result is None or _cap_reached(provider):
            if not_run is not None:
                not_run.extend(doc_ids[position:])
            break
        results.append(result)
    return results


def _cap_reached(provider: LLMProvider) -> bool:
    return bool(getattr(provider, "cap_reached", False))


def _run_document(
    path: str,
    doc_id: str,
    provider: LLMProvider,
    reader: Callable[[str], ReaderOutput],
    chart: Sequence[Dict[str, Any]],
) -> DocResult:
    try:
        reader_output = reader(path)
    except ReaderError as exc:
        return DocResult(doc_id, error={"stage": "reader", "message": str(exc)})
    try:
        extraction = extract_invoice(reader_output, provider)
    except ExtractionError as exc:
        return DocResult(doc_id, error=exc.to_dict())
    codes = code_lines(extraction.line_items, chart, provider, doc_id, vendor_name=extraction.vendor_name)
    return DocResult(doc_id, extraction=extraction, gl_codes=codes)


def run_eval(
    provider: LLMProvider,
    reader: Callable[[str], ReaderOutput] = read_document,
    corpus_dir: str = CORPUS_DIR,
) -> Dict[str, Any]:
    """Run the pipeline over the corpus and score it. Returns the scored structure plus the run's banner.

    A live run also carries `live` (model, effort, spend) and `not_run` (documents the spend cap kept from running).
    """
    banner = banner_for(provider)
    not_run: List[str] = []
    results = run_pipeline(list_documents(corpus_dir), provider, reader, not_run=not_run)
    scored = scoring.score_results(results, corpus_dir)
    scored["banner"] = banner
    scored["provider"] = type(provider).__name__
    scored["mode"] = "live" if is_live(provider) else "fixture"
    if is_live(provider):
        scored["not_run"] = not_run
        scored["live"] = _live_info(provider)
    return scored


def _live_info(provider: AnthropicProvider) -> Dict[str, Any]:
    tracker = provider.tracker
    return {
        "model": provider.model,
        "effort": provider.effort,
        "max_tokens": provider.max_tokens,
        "ocr_langs": resolve_ocr_langs(),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calls": tracker.run_calls,
        "run_spent_usd": format(tracker.run_spent_usd, "f"),
        "ledger_total_usd": format(tracker.total_usd, "f"),
        "cap_usd": format(tracker.cap_usd, "f"),
        "run_cap_usd": None if tracker.run_cap_usd is None else format(tracker.run_cap_usd, "f"),
    }


# --- report ------------------------------------------------------------------------------------------------------


def _cell(count: int, n: int) -> str:
    if n == 0:
        return "-"
    return f"{count}/{n} ({100 * count / n:.1f}%)"


def _pct(rate: Optional[float]) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


def _count(count: int, n: int) -> str:
    """A GL result as a bare count. GL results are never printed as percentages: the labels and fixtures share an
    author, so a GL rate would invite quoting it as accuracy."""
    return f"{count}/{n}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    widths = [max(len(str(r[c])) for r in [headers, *rows]) for c in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(str(v).ljust(widths[c]) for c, v in enumerate(row)).rstrip()

    return [fmt(headers), "  ".join("-" * w for w in widths), *(fmt(r) for r in rows)]


def _usd(amount: str) -> str:
    return f"${Decimal(amount):.4f}"


def _live_header(scored: Dict[str, Any]) -> List[str]:
    """Lines that open the live report body: the incomplete-run notice (if any) and the live section heading."""
    info = scored["live"]
    ran = scored["summary"]["overall"]["documents"]
    not_run = scored["not_run"]
    out: List[str] = []
    if not_run:
        out.append(
            f"INCOMPLETE RUN — the spend cap stopped the run after {ran} of {ran + len(not_run)} documents; every figure "
            f"below covers only those {ran}. Not run: {', '.join(not_run)}"
        )
        out.append("")
    out.append(
        f"========== LIVE MODEL ACCURACY — {info['model']}, effort {info['effort']}, {info['run_at']}: "
        "scored against reference answers the model never saw =========="
    )
    return out


def _live_cost_section(scored: Dict[str, Any]) -> List[str]:
    info = scored["live"]
    failed = [d["extraction_error"] for d in scored["documents"] if d["extraction_error"]]
    by_stage = Counter(err.get("stage", "unknown") for err in failed)
    run_cap = f" (run cap {_usd(info['run_cap_usd'])})" if info["run_cap_usd"] else ""
    out = [
        "5. LIVE RUN COST AND FAILURES",
        f"   Model calls this run: {info['calls']}   Spend this run: {_usd(info['run_spent_usd'])}{run_cap}   "
        f"Ledger total: {_usd(info['ledger_total_usd'])} of {_usd(info['cap_usd'])} cap",
        "   Reader and extraction failures by stage: "
        + (", ".join(f"{stage}: {n}" for stage, n in sorted(by_stage.items())) if by_stage else "none"),
    ]
    messages = list(dict.fromkeys(str(err.get("message", "")) for err in failed))[:3]
    out += [f"   failure message: {m}" for m in messages]
    out.append("")
    return out


def render_report(scored: Dict[str, Any]) -> str:
    summary = scored["summary"]
    overall = summary["overall"]
    live = scored.get("mode") == "live"
    out: List[str] = [scored["banner"], ""]
    if live:
        out += _live_header(scored)
    out.append(f"Provider: {scored['provider']}   Documents: {overall['documents']}   "
               f"Extraction/reader failures: {overall['extraction_errors']}")
    failed = [d["doc_id"] for d in scored["documents"] if d["extraction_error"]]
    if failed:
        out.append("Failed documents (every field scored as a miss): " + ", ".join(failed))
    out.append("")

    out.append("1. HEADER FIELDS — match against ground truth, scorable cells only (ground truth non-null)")
    out.append("   strict = case/whitespace-insensitive, accents kept; lenient = accents folded (separate column, never merged)")
    rows = []
    for name in (*scoring.HEADER_FIELDS, scoring.ALL_HEADER):
        c = overall["header"][name] if name in overall["header"] else overall[name]
        rows.append([name, _cell(c["strict"], c["n"]), _cell(c["lenient"], c["n"]), str(c["spurious"])])
    out += _table(["field", "strict", "lenient (accent-folded)", "value where truth is null"], rows)
    out.append("")

    out.append("2. LINE ITEMS — cells of each ground-truth line, matched by description/position")
    rows = []
    for name in (*scoring.LINE_FIELDS, scoring.ALL_LINE):
        c = overall["lines"][name] if name in overall["lines"] else overall[name]
        rows.append([name, _cell(c["strict"], c["n"]), _cell(c["lenient"], c["n"])])
    out += _table(["cell", "strict", "lenient (accent-folded)"], rows)
    lc = overall["line_counts"]
    out.append(f"   ground-truth lines: {lc['truth']}   extra extracted lines with no ground-truth partner: {lc['extra']}")
    out.append("")

    out.append("3. BY LANGUAGE AND BY VARIANT")
    rows = []
    for key, label in [("lang:en", "en"), ("lang:es", "es"), ("variant:pdf", "pdf"),
                       ("variant:scan", "scan"), ("variant:photo", "photo"), ("overall", "all")]:
        g = summary[key]
        rows.append([
            label,
            str(g["documents"]),
            _cell(g[scoring.ALL_HEADER]["strict"], g[scoring.ALL_HEADER]["n"]),
            _cell(g[scoring.ALL_HEADER]["lenient"], g[scoring.ALL_HEADER]["n"]),
            _cell(g[scoring.ALL_LINE]["strict"], g[scoring.ALL_LINE]["n"]),
            _count(g["gl"]["correct"], g["gl"]["scorable"]),
            str(g["extraction_errors"]),
        ])
    out += _table(
        ["group", "docs", "header strict", "header lenient", "line cells strict", "GL (LLM code = label)", "failures"],
        rows,
    )
    out.append("")

    gl = overall["gl"]
    out.append("4. GL CODING — extracted lines coded by the provider, scored on lines with a non-null label only")
    llm_coded = gl["correct"] + gl["incorrect"]
    rows = [
        ["Lines with a non-null label (scorable)", str(gl["scorable"])],
        ["  LLM code equals label", str(gl["correct"])],
        ["  LLM code differs from label", str(gl["incorrect"])],
        ["  fell back to the keyword coder (not scored)", str(gl["fallback"])],
        ["  no extracted line for the label", str(gl["no_line"])],
        ["Unscorable (no fitting account)", str(gl["unscorable"])],
        ["LLM code matched the label", _count(gl["correct"], gl["scorable"]) + " scorable lines matched"],
        ["LLM code matched, of the lines it coded", _count(gl["correct"], llm_coded) + " LLM-coded lines matched"],
    ]
    out += _table(["row", "lines"], rows)
    fallbacks = [f"{d['doc_id']}#{r['line']}" for d in scored["documents"] for r in d["gl"] if r["status"] == "fallback"]
    out.append("   keyword-coder fallback lines: " + (", ".join(fallbacks) if fallbacks else "none"))
    out.append("")
    if live:
        out += _live_cost_section(scored)
    out.append(scored["banner"])
    return "\n".join(out) + "\n"


def write_reports(scored: Dict[str, Any], out_dir: str) -> List[str]:
    """Write the text and JSON reports; both start with the banner."""
    os.makedirs(out_dir, exist_ok=True)
    live = scored.get("mode") == "live"
    txt_path = os.path.join(out_dir, LIVE_REPORT_TXT if live else REPORT_TXT)
    json_path = os.path.join(out_dir, LIVE_REPORT_JSON if live else REPORT_JSON)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_report(scored))
    # The banner is the first key so the file opens with the disclosure, as the text report does.
    ordered = {"banner": scored["banner"], **{k: v for k, v in scored.items() if k != "banner"}}
    # GL results are counts only in every rendering, so the GL match rate is left out of the JSON as well.
    ordered["summary"] = {
        group: {k: v for k, v in values.items() if k != "gl_match_rate"} for group, values in scored["summary"].items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=1, ensure_ascii=False, default=str)
    return [txt_path, json_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extraction + GL coding eval over the synthetic corpus: an offline fixture plumbing check by default, "
        "or a separately labelled live-model run with --provider anthropic."
    )
    parser.add_argument("--provider", default=None, help="LLM provider name (default: AP_LLM_PROVIDER or fixture)")
    parser.add_argument("--fixtures-dir", default=None, help="directory holding extract/ and gl/ fixture folders")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="directory for the report files (default: eval/out)")
    parser.add_argument(
        "--max-usd", type=float, default=None,
        help="cap on this run's live-model spend in USD; required with a live provider, refused otherwise",
    )
    args = parser.parse_args(argv)

    # Only configuration errors are refusals (exit 2); a failure while running or scoring is a bug and must crash.
    try:
        provider = get_provider(args.provider, args.fixtures_dir, max_usd=args.max_usd)
        banner_for(provider)
        if is_live(provider) and args.max_usd is None:
            raise ValueError("--max-usd is required with a live provider: give this run its own spend cap")
        if not is_live(provider) and args.max_usd is not None:
            raise ValueError("--max-usd applies only to a live provider")
    except (NotImplementedError, ValueError) as exc:
        print(f"extraction eval not run: {exc}", file=sys.stderr)
        return 2
    scored = run_eval(provider)
    report = render_report(scored)
    paths = write_reports(scored, args.out)
    sys.stdout.write(report)
    print("Wrote: " + ", ".join(paths))
    if scored.get("mode") == "live":
        info = scored["live"]
        print(
            f"Spend this run: {_usd(info['run_spent_usd'])} in {info['calls']} calls; "
            f"ledger total {_usd(info['ledger_total_usd'])} of {_usd(info['cap_usd'])} cap"
        )
        if scored["not_run"]:
            print(f"INCOMPLETE: the spend cap stopped the run; {len(scored['not_run'])} documents were not run.", file=sys.stderr)
            return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    sys.exit(main())

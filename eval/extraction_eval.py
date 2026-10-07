# ABOUTME: Offline end-to-end eval: reader -> LLM extraction -> GL coding, then scored by extraction_scoring.
# ABOUTME: Fixture-backed plumbing check, never a model-accuracy measurement; every report opens with a disclosure banner.
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Make `python eval/extraction_eval.py` find the sibling scorer and the package without relying on PYTHONPATH.
for _path in (_HERE, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import extraction_scoring as scoring  # noqa: E402
from ap_invoice_processor.llm.extraction import ExtractedInvoice, ExtractionError, extract_invoice  # noqa: E402
from ap_invoice_processor.llm.gl import GLCode, code_lines, load_chart  # noqa: E402
from ap_invoice_processor.llm.provider import FixtureProvider, LLMProvider, get_provider  # noqa: E402
from ap_invoice_processor.reader import ReaderError, ReaderOutput, read_document  # noqa: E402

CORPUS_DIR = scoring.CORPUS_DIR
DEFAULT_OUT_DIR = os.path.join(_HERE, "out")
REPORT_TXT = "extraction_eval.txt"
REPORT_JSON = "extraction_eval.json"

BANNER = (
    "PLUMBING CHECK — offline, fixture-backed. Fixtures are hand-authored from reader text; no model ran. "
    "These numbers validate the scoring pipeline and wiring only and are NOT model accuracy. "
    "GL fixture/label agreement is not independent (same author). "
    "Spanish scans/photos OCR'd with -l eng."
)


@dataclass
class DocResult:
    """What the pipeline produced for one document. It holds no reference data of any kind."""

    doc_id: str
    extraction: Optional[ExtractedInvoice] = None
    gl_codes: List[GLCode] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None


def banner_for(provider: LLMProvider) -> str:
    """The disclosure banner for a run. Only the fixture provider has an approved wording."""
    if isinstance(provider, FixtureProvider):
        return BANNER
    raise NotImplementedError(
        "no disclosure banner is defined for a live provider; its wording must be agreed before a live run is reported"
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
) -> List[DocResult]:
    """Read, extract and GL-code each document. Extraction receives only the ReaderOutput; the GL coder receives
    only the extracted lines, the chart and the extracted vendor name."""
    chart = chart if chart is not None else load_chart()
    results: List[DocResult] = []
    for path in doc_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        try:
            reader_output = reader(path)
        except ReaderError as exc:
            results.append(DocResult(doc_id, error={"stage": "reader", "message": str(exc)}))
            continue
        try:
            extraction = extract_invoice(reader_output, provider)
        except ExtractionError as exc:
            results.append(DocResult(doc_id, error=exc.to_dict()))
            continue
        codes = code_lines(
            extraction.line_items, chart, provider, doc_id, vendor_name=extraction.vendor_name
        )
        results.append(DocResult(doc_id, extraction=extraction, gl_codes=codes))
    return results


def run_eval(
    provider: LLMProvider,
    reader: Callable[[str], ReaderOutput] = read_document,
    corpus_dir: str = CORPUS_DIR,
) -> Dict[str, Any]:
    """Run the pipeline over the corpus and score it. Returns the scored structure plus the run's banner."""
    banner = banner_for(provider)
    results = run_pipeline(list_documents(corpus_dir), provider, reader)
    scored = scoring.score_results(results, corpus_dir)
    scored["banner"] = banner
    scored["provider"] = type(provider).__name__
    return scored


# --- report ------------------------------------------------------------------------------------------------------


def _cell(count: int, n: int) -> str:
    if n == 0:
        return "-"
    return f"{count}/{n} ({100 * count / n:.1f}%)"


def _pct(rate: Optional[float]) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    widths = [max(len(str(r[c])) for r in [headers, *rows]) for c in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(str(v).ljust(widths[c]) for c, v in enumerate(row)).rstrip()

    return [fmt(headers), "  ".join("-" * w for w in widths), *(fmt(r) for r in rows)]


def render_report(scored: Dict[str, Any]) -> str:
    summary = scored["summary"]
    overall = summary["overall"]
    out: List[str] = [scored["banner"], ""]
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
            _cell(g["gl"]["correct"], g["gl"]["scorable"]),
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
        ["LLM equals label / scorable lines", _cell(gl["correct"], gl["scorable"])],
        ["LLM equals label / LLM-coded lines", _cell(gl["correct"], llm_coded)],
    ]
    out += _table(["row", "lines"], rows)
    fallbacks = [f"{d['doc_id']}#{r['line']}" for d in scored["documents"] for r in d["gl"] if r["status"] == "fallback"]
    out.append("   keyword-coder fallback lines: " + (", ".join(fallbacks) if fallbacks else "none"))
    out.append("")
    out.append(scored["banner"])
    return "\n".join(out) + "\n"


def write_reports(scored: Dict[str, Any], out_dir: str) -> List[str]:
    """Write the text and JSON reports; both start with the banner."""
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, REPORT_TXT)
    json_path = os.path.join(out_dir, REPORT_JSON)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_report(scored))
    # The banner is the first key so the file opens with the disclosure, as the text report does.
    ordered = {"banner": scored["banner"], **{k: v for k, v in scored.items() if k != "banner"}}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=1, ensure_ascii=False, default=str)
    return [txt_path, json_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline extraction + GL coding plumbing check over the synthetic corpus.")
    parser.add_argument("--provider", default=None, help="LLM provider name (default: AP_LLM_PROVIDER or fixture)")
    parser.add_argument("--fixtures-dir", default=None, help="directory holding extract/ and gl/ fixture folders")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="directory for the report files (default: eval/out)")
    args = parser.parse_args(argv)

    try:
        provider = get_provider(args.provider, args.fixtures_dir)
        scored = run_eval(provider)
    except (NotImplementedError, ValueError) as exc:
        print(f"extraction eval not run: {exc}", file=sys.stderr)
        return 2
    report = render_report(scored)
    paths = write_reports(scored, args.out)
    sys.stdout.write(report)
    print("Wrote: " + ", ".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())

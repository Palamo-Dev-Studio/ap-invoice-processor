# ABOUTME: Scorer for the offline extraction eval: the only eval module that loads ground truth and GL labels.
# ABOUTME: Compares extracted fields and GL codes with them (strict and accent-folded columns kept separate) and aggregates counts.
import json
import os
import unicodedata
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus"
)

HEADER_TEXT_FIELDS = ("vendor_name", "invoice_number", "po_number")
HEADER_DATE_FIELDS = ("invoice_date", "due_date")
HEADER_AMOUNT_FIELDS = ("subtotal", "tax", "total")
HEADER_FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "po_number",
    "subtotal",
    "tax",
    "total",
)
LINE_FIELDS = ("description", "quantity", "unit_price", "amount")
LANGUAGES = ("en", "es")
VARIANTS = ("pdf", "scan", "photo")
# Row keys for the aggregate rows of the field tables.
ALL_HEADER = "ALL header fields"
ALL_LINE = "ALL line-item cells"


# --- normalisation -----------------------------------------------------------------------------------------------


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


def strict_text(value: Any) -> Optional[str]:
    """Case- and whitespace-insensitive form of a text value; accents are kept. None stays None."""
    return None if value is None else _squash(str(value))


def lenient_text(value: Any) -> Optional[str]:
    """The strict form with accents folded away. Used only for the separate lenient column."""
    folded = strict_text(value)
    if folded is None:
        return None
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def money(value: Any) -> Optional[Decimal]:
    """An amount as a Decimal rounded half-up to cents; None stays None."""
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _norm_date(value: Any) -> Optional[str]:
    return None if value is None else str(value).strip()


def _norm_currency(value: Any) -> Optional[str]:
    return None if value is None else str(value).strip().upper()


def _norm(field: str, value: Any, lenient: bool) -> Any:
    if field in HEADER_TEXT_FIELDS or field == "description":
        return lenient_text(value) if lenient else strict_text(value)
    if field in HEADER_DATE_FIELDS:
        return _norm_date(value)
    if field == "currency":
        return _norm_currency(value)
    return money(value)


def fields_match(field: str, extracted: Any, expected: Any) -> Tuple[bool, bool]:
    """Return (strict match, lenient match) for one non-null expected value. An absent extraction never matches."""
    if extracted is None:
        return False, False
    return (
        _norm(field, extracted, False) == _norm(field, expected, False),
        _norm(field, extracted, True) == _norm(field, expected, True),
    )


# --- ground truth and labels (loaded here and nowhere else in the eval) -----------------------------------------


def load_ground_truth(doc_id: str, corpus_dir: str = CORPUS_DIR) -> Dict[str, Any]:
    with open(os.path.join(corpus_dir, "ground_truth", f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def load_gl_labels(corpus_dir: str = CORPUS_DIR) -> Dict[str, List[Dict[str, Any]]]:
    with open(os.path.join(corpus_dir, "gl_labels.json"), encoding="utf-8") as f:
        return json.load(f)["labels"]


# --- line matching -----------------------------------------------------------------------------------------------


def match_lines(extracted: Sequence[Any], truth: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """Map extracted line index -> ground-truth line index.

    Pass 1 pairs lines at the same position whose descriptions agree (accents folded, so OCR damage does not
    break the pairing); pass 2 pairs the remaining lines whose descriptions agree wherever they sit; pass 3 pairs
    whatever is left by position. A line the pairing gets wrong is scored as wrong, not hidden.
    """
    pairs: Dict[int, int] = {}
    used_truth = set()

    def same(i: int, j: int) -> bool:
        return lenient_text(extracted[i].description) == lenient_text(truth[j]["description"])

    for i in range(min(len(extracted), len(truth))):
        if same(i, i):
            pairs[i] = i
            used_truth.add(i)
    for i in range(len(extracted)):
        if i in pairs:
            continue
        for j in range(len(truth)):
            if j not in used_truth and same(i, j):
                pairs[i] = j
                used_truth.add(j)
                break
    for i in range(len(extracted)):
        if i not in pairs and i < len(truth) and i not in used_truth:
            pairs[i] = i
            used_truth.add(i)
    return pairs


# --- per-document scoring ----------------------------------------------------------------------------------------


def score_document(result: Any, ground_truth: Dict[str, Any], labels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one pipeline result (extraction + GL codes) against its ground truth and GL labels.

    Cells are scorable only where the ground truth holds a value. A value returned where the ground truth is
    null is counted separately as spurious. The strict result and the accent-folded result are kept apart.
    """
    extraction = result.extraction
    header: Dict[str, Dict[str, Any]] = {}
    for field in HEADER_FIELDS:
        expected = ground_truth.get(field)
        got = getattr(extraction, field, None) if extraction is not None else None
        if expected is None:
            header[field] = {"scorable": False, "spurious": got is not None, "got": got, "expected": None}
            continue
        strict, lenient = fields_match(field, got, expected)
        header[field] = {
            "scorable": True,
            "strict": strict,
            "lenient": lenient,
            "spurious": False,
            "got": got,
            "expected": expected,
        }

    truth_lines = ground_truth["line_items"]
    extracted_lines = list(extraction.line_items) if extraction is not None else []
    pairs = match_lines(extracted_lines, truth_lines)
    by_truth = {j: i for i, j in pairs.items()}
    lines: List[Dict[str, Any]] = []
    for j, truth_line in enumerate(truth_lines):
        i = by_truth.get(j)
        cells: Dict[str, Dict[str, Any]] = {}
        for field in LINE_FIELDS:
            expected = truth_line.get(field)
            got = getattr(extracted_lines[i], field) if i is not None else None
            strict, lenient = fields_match(field, got, expected)
            cells[field] = {"strict": strict, "lenient": lenient, "got": got, "expected": expected}
        lines.append({"truth_line": j, "extracted_line": i, "cells": cells})

    gl = _score_gl(result, labels, by_truth)
    return {
        "doc_id": result.doc_id,
        "language": ground_truth["language"],
        "variant": ground_truth["variant"],
        "extraction_error": result.error,
        "header": header,
        "lines": lines,
        "extra_extracted_lines": len(extracted_lines) - len(pairs),
        "truth_line_count": len(truth_lines),
        "gl": gl,
    }


def _score_gl(result: Any, labels: Sequence[Dict[str, Any]], by_truth: Dict[int, int]) -> List[Dict[str, Any]]:
    """One record per label line. status is one of: correct, incorrect, fallback, no_line, unscorable."""
    records = []
    for label in labels:
        j = label["line"]
        expected = label["account"]
        i = by_truth.get(j)
        code = result.gl_codes[i] if i is not None and i < len(result.gl_codes) else None
        record = {"line": j, "expected": expected, "got": code.account if code else None}
        if expected is None:
            record["status"] = "unscorable"
        elif code is None:
            record["status"] = "no_line"
        elif code.source != "llm":
            record["status"] = "fallback"
        elif code.account == expected:
            record["status"] = "correct"
        else:
            record["status"] = "incorrect"
        records.append(record)
    return records


# --- aggregation -------------------------------------------------------------------------------------------------


def _blank_counts() -> Dict[str, int]:
    return {"n": 0, "strict": 0, "lenient": 0, "spurious": 0}


def _blank_gl() -> Dict[str, int]:
    return {
        "scorable": 0,
        "correct": 0,
        "incorrect": 0,
        "fallback": 0,
        "no_line": 0,
        "unscorable": 0,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def aggregate(documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll per-document scores up overall and per language and per variant."""
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": list(documents)}
    for lang in LANGUAGES:
        groups[f"lang:{lang}"] = [d for d in documents if d["language"] == lang]
    for variant in VARIANTS:
        groups[f"variant:{variant}"] = [d for d in documents if d["variant"] == variant]

    out: Dict[str, Any] = {}
    for name, docs in groups.items():
        header: Dict[str, Dict[str, int]] = defaultdict(_blank_counts)
        lines: Dict[str, Dict[str, int]] = defaultdict(_blank_counts)
        gl = _blank_gl()
        line_counts = {"truth": 0, "extra": 0}
        for d in docs:
            for field, cell in d["header"].items():
                row = header[field]
                if cell["scorable"]:
                    row["n"] += 1
                    row["strict"] += cell["strict"]
                    row["lenient"] += cell["lenient"]
                elif cell["spurious"]:
                    row["spurious"] += 1
            line_counts["truth"] += d["truth_line_count"]
            line_counts["extra"] += d["extra_extracted_lines"]
            for line in d["lines"]:
                for field, cell in line["cells"].items():
                    row = lines[field]
                    row["n"] += 1
                    row["strict"] += cell["strict"]
                    row["lenient"] += cell["lenient"]
            for record in d["gl"]:
                gl[record["status"]] += 1
                if record["status"] != "unscorable":
                    gl["scorable"] += 1
        all_header = _blank_counts()
        for row in header.values():
            for key in all_header:
                all_header[key] += row[key]
        all_lines = _blank_counts()
        for row in lines.values():
            for key in all_lines:
                all_lines[key] += row[key]
        everything = {key: all_header[key] + all_lines[key] for key in all_header}
        out[name] = {
            "documents": len(docs),
            "extraction_errors": sum(1 for d in docs if d["extraction_error"]),
            "header": {field: header[field] for field in HEADER_FIELDS},
            ALL_HEADER: all_header,
            "lines": {field: lines[field] for field in LINE_FIELDS},
            ALL_LINE: all_lines,
            "line_counts": line_counts,
            "gl": gl,
            "extraction_strict_rate": _rate(everything["strict"], everything["n"]),
            "extraction_lenient_rate": _rate(everything["lenient"], everything["n"]),
            "header_strict_rate": _rate(all_header["strict"], all_header["n"]),
            "line_strict_rate": _rate(all_lines["strict"], all_lines["n"]),
            "gl_match_rate": _rate(gl["correct"], gl["scorable"]),
        }
    return out


def score_results(results: Sequence[Any], corpus_dir: str = CORPUS_DIR) -> Dict[str, Any]:
    """Score every pipeline result and aggregate. This is the single entry point that reads ground truth and labels."""
    all_labels = load_gl_labels(corpus_dir)
    documents = [
        score_document(r, load_ground_truth(r.doc_id, corpus_dir), all_labels.get(r.doc_id, []))
        for r in results
    ]
    return {"documents": documents, "summary": aggregate(documents)}

# ABOUTME: Tests for the document-reading layer against the synthetic corpus (pdftotext, Tesseract, error paths).
# ABOUTME: Also checks every ground-truth file for internal arithmetic consistency to catch corpus generator bugs.
import glob
import json
import os
import subprocess
from decimal import Decimal

import pytest

from ap_invoice_processor import reader
from ap_invoice_processor.reader import ReaderError, read_document

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus")
GT_FILES = sorted(glob.glob(os.path.join(CORPUS, "ground_truth", "*.json")))
PDF_IDS = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CORPUS, "pdf", "*.pdf")))


def _gt(doc_id):
    with open(os.path.join(CORPUS, "ground_truth", f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_corpus_present():
    assert len(PDF_IDS) == 30
    assert len(GT_FILES) >= 40


@pytest.mark.parametrize("doc_id", PDF_IDS)
def test_pdf_text_contains_vendor_and_total(doc_id):
    gt = _gt(doc_id)
    out = read_document(os.path.join(CORPUS, "pdf", f"{doc_id}.pdf"))
    assert out.doc_id == doc_id
    assert out.method == "pdftotext"
    assert out.ocr_lang is None
    assert gt["vendor_name"] in out.text
    assert gt["total_display"] in out.text
    assert gt["invoice_number"] in out.text


def test_scan_variant_uses_tesseract_english():
    # en-005-scan is a deterministic fixture whose OCR output carries the vendor name and total.
    gt = _gt("en-005-scan")
    out = read_document(os.path.join(CORPUS, "images", "en-005-scan.png"))
    assert out.doc_id == "en-005-scan"
    assert out.method == "tesseract"
    assert out.ocr_lang == "eng"
    assert gt["vendor_name"] in out.text
    assert gt["total_display"] in out.text


def test_empty_pdf_text_falls_back_to_ocr(monkeypatch):
    gt = _gt("en-001")
    monkeypatch.setattr(reader, "_pdf_text", lambda path: "  \n")
    out = read_document(os.path.join(CORPUS, "pdf", "en-001.pdf"))
    assert out.method == "tesseract"
    assert out.ocr_lang == "eng"
    assert gt["vendor_name"] in out.text


def test_timeout_raises_reader_error(monkeypatch):
    def boom(cmd, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(reader.subprocess, "run", boom)
    with pytest.raises(ReaderError, match="timed out"):
        read_document(os.path.join(CORPUS, "pdf", "en-001.pdf"))


def test_missing_binary_raises_reader_error(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(reader.subprocess, "run", missing)
    with pytest.raises(ReaderError, match="not installed"):
        read_document(os.path.join(CORPUS, "images", "en-005-scan.png"))


def test_nonzero_exit_raises_reader_error(monkeypatch):
    def failed(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad file")

    monkeypatch.setattr(reader.subprocess, "run", failed)
    with pytest.raises(ReaderError, match="exited with code 1: bad file"):
        read_document(os.path.join(CORPUS, "pdf", "en-001.pdf"))


def test_missing_and_unsupported_files_raise_reader_error(tmp_path):
    with pytest.raises(ReaderError, match="not found"):
        read_document(str(tmp_path / "nope.pdf"))
    odd = tmp_path / "invoice.docx"
    odd.write_text("x")
    with pytest.raises(ReaderError, match="unsupported"):
        read_document(str(odd))


@pytest.mark.parametrize("gt_path", GT_FILES, ids=lambda p: os.path.basename(p))
def test_ground_truth_arithmetic(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    def d(x):
        return Decimal(str(x))

    assert gt["language"] in ("en", "es")
    assert gt["variant"] in ("pdf", "scan", "photo")
    assert gt["line_items"], "every invoice has at least one line item"
    for li in gt["line_items"]:
        assert d(li["quantity"]) * d(li["unit_price"]) == d(li["amount"])
    assert sum(d(li["amount"]) for li in gt["line_items"]) == d(gt["subtotal"])
    assert d(gt["subtotal"]) + d(gt["tax"]) == d(gt["total"])
    if gt["variant"] != "pdf":
        assert gt["ocr_lang"] == "eng"
    assert os.path.isfile(os.path.join(CORPUS, gt["source_file"]))

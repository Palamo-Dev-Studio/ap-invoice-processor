# ABOUTME: Tests for the document-reading layer against the synthetic corpus (pdftotext, Tesseract, error paths).
# ABOUTME: Also checks every ground-truth file for internal arithmetic consistency to catch corpus generator bugs.
import glob
import json
import os
import re
import shutil
import subprocess
from decimal import Decimal

import pytest

from ap_invoice_processor import reader
from ap_invoice_processor.reader import ReaderError, read_document

needs_binaries = pytest.mark.skipif(
    not (shutil.which("pdftotext") and shutil.which("tesseract") and shutil.which("pdftoppm")),
    reason="poppler-utils/tesseract not installed",
)

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus")
GT_FILES = sorted(glob.glob(os.path.join(CORPUS, "ground_truth", "*.json")))
PDF_IDS = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CORPUS, "pdf", "*.pdf")))


def _gt(doc_id):
    with open(os.path.join(CORPUS, "ground_truth", f"{doc_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_corpus_present():
    assert len(PDF_IDS) == 30
    assert len(GT_FILES) >= 40


@needs_binaries
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


IMAGE_IDS = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CORPUS, "images", "*.png")))


@needs_binaries
@pytest.mark.parametrize("doc_id", IMAGE_IDS)
def test_image_variant_ocr_recovers_invoice_number_and_total(doc_id):
    # Covers all layouts: the twocol sidebar is lost by Tesseract's default thresholding alone.
    gt = _gt(doc_id)
    out = read_document(os.path.join(CORPUS, "images", f"{doc_id}.png"))
    assert out.doc_id == doc_id
    assert out.method == "tesseract"
    assert out.ocr_lang == "eng+spa"
    assert gt["invoice_number"] in out.text
    assert gt["total_display"] in out.text


def test_corpus_has_ten_image_variants():
    assert len(IMAGE_IDS) == 10


def test_tesseract_argv_runs_both_thresholding_modes_with_the_given_languages(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(cmd, 0, stdout=f"out{len(calls)}", stderr="")

    monkeypatch.setattr(reader.subprocess, "run", fake_run)
    text = reader._ocr_image("page.png", "eng+spa")
    assert len(calls) == 2
    for cmd in calls:
        assert cmd[0] == "tesseract"
        assert cmd[cmd.index("-l") + 1] == "eng+spa"
    assert "thresholding_method=2" not in calls[0]
    assert calls[1][-2:] == ["-c", "thresholding_method=2"]
    assert text == "out1\nout2"


@needs_binaries
def test_empty_pdf_text_falls_back_to_ocr(monkeypatch):
    gt = _gt("en-001")
    monkeypatch.setattr(reader, "_pdf_text", lambda path: "  \n")
    out = read_document(os.path.join(CORPUS, "pdf", "en-001.pdf"))
    assert out.method == "tesseract"
    assert out.ocr_lang == "eng+spa"
    assert gt["vendor_name"] in out.text


@pytest.mark.parametrize("name, tool", [("-opts.pdf", "pdftotext"), ("-opts.png", "tesseract")])
def test_relative_path_starting_with_dash_reaches_the_tool_as_an_absolute_path(tmp_path, monkeypatch, name, tool):
    (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="plenty of invoice text here", stderr="")

    monkeypatch.setattr(reader.subprocess, "run", fake_run)
    out = read_document(name)
    assert out.doc_id == os.path.splitext(name)[0]
    assert calls and calls[0][0] == tool
    paths = [arg for arg in calls[0] if arg.endswith(name)]
    assert len(paths) == 1 and os.path.isabs(paths[0])
    assert not any(arg.startswith("-opts") for arg in calls[0])


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


# --- OCR languages -----------------------------------------------------------------------------------------------

FIXTURES_OCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ocr")


def _installed_tesseract_langs():
    if not shutil.which("tesseract"):
        return set()
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=False).stdout
    return {line.strip() for line in out.splitlines()[1:]}


def _needs_langs(*langs):
    missing = [lang for lang in langs if lang not in _installed_tesseract_langs()]
    return pytest.mark.skipif(bool(missing), reason=f"tesseract language data not installed: {missing}")


def test_default_ocr_languages_are_english_plus_spanish(monkeypatch):
    monkeypatch.delenv(reader.OCR_LANGS_ENV, raising=False)
    assert reader.resolve_ocr_langs() == "eng+spa"
    assert reader.DEFAULT_OCR_LANGS == ("eng", "spa")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("eng", "eng"),
        ("eng+spa+chi_sim", "eng+spa+chi_sim"),
        ("spa, eng", "spa+eng"),
        ("eng,chi_tra", "eng+chi_tra"),
        (["eng", "chi_sim"], "eng+chi_sim"),
        (("eng", "spa", "eng"), "eng+spa"),
    ],
)
def test_resolve_ocr_langs_accepts_the_supported_set_in_order_without_duplicates(value, expected):
    assert reader.resolve_ocr_langs(value) == expected


@pytest.mark.parametrize("value", ["", "  ", [], "fra", "eng+deu", "eng+spa;x", "-c", "../eng", "eng spa x", "ENG"])
def test_resolve_ocr_langs_rejects_anything_outside_the_supported_set(value):
    with pytest.raises(ReaderError, match="OCR language"):
        reader.resolve_ocr_langs(value)


def test_ocr_langs_env_sets_the_default_and_an_argument_beats_it(monkeypatch):
    monkeypatch.setenv(reader.OCR_LANGS_ENV, "eng+chi_sim")
    assert reader.resolve_ocr_langs() == "eng+chi_sim"
    assert reader.resolve_ocr_langs("spa") == "spa"


def test_a_bad_env_value_is_a_reader_error_not_a_silent_default(monkeypatch):
    monkeypatch.setenv(reader.OCR_LANGS_ENV, "klingon")
    with pytest.raises(ReaderError, match="OCR language"):
        reader.resolve_ocr_langs()


def test_read_document_passes_the_languages_to_every_tesseract_call_and_records_them(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"x")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="text", stderr="")

    monkeypatch.setattr(reader.subprocess, "run", fake_run)
    monkeypatch.delenv(reader.OCR_LANGS_ENV, raising=False)

    out = read_document(str(image))
    assert out.ocr_lang == "eng+spa"
    assert [cmd[cmd.index("-l") + 1] for cmd in calls] == ["eng+spa", "eng+spa"]

    calls.clear()
    out = read_document(str(image), ocr_langs=["eng", "chi_sim"])
    assert out.ocr_lang == "eng+chi_sim"
    assert [cmd[cmd.index("-l") + 1] for cmd in calls] == ["eng+chi_sim", "eng+chi_sim"]


def test_pdf_ocr_fallback_uses_the_requested_languages(monkeypatch, tmp_path):
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"x")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "pdftoppm":
            open(cmd[-1] + "-1.png", "wb").close()
        if cmd[0] == "pdfinfo":
            return subprocess.CompletedProcess(cmd, 0, stdout="Pages: 1\nPage    1 size:  612 x 792 pts (letter)\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(reader.subprocess, "run", fake_run)
    out = read_document(str(pdf), ocr_langs="spa")
    assert out.method == "tesseract" and out.ocr_lang == "spa"
    tesseract_calls = [c for c in calls if c[0] == "tesseract"]
    assert len(tesseract_calls) == 2
    assert all(c[c.index("-l") + 1] == "spa" for c in tesseract_calls)


def test_an_invalid_language_fails_before_any_tool_runs(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"x")

    def boom(cmd, **kwargs):
        raise AssertionError("a tool ran")

    monkeypatch.setattr(reader.subprocess, "run", boom)
    with pytest.raises(ReaderError, match="OCR language"):
        read_document(str(image), ocr_langs="fra")


ES_IMAGE_IDS = [i for i in IMAGE_IDS if i.startswith("es-")]


def _accented_words_recovered(doc_id, langs):
    """How many accented words of the ground truth appear verbatim in the OCR text (a quick count, not accuracy)."""
    gt = _gt(doc_id)

    def strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from strings(v)

    accented = {w for s in strings(gt) for w in re.findall(r"\w+", s) if re.search(r"[áéíóúñüÁÉÍÓÚÑÜ]", w)}
    text = read_document(os.path.join(CORPUS, "images", f"{doc_id}.png"), ocr_langs=langs).text.casefold()
    return sum(1 for w in accented if w.casefold() in text), len(accented)


@needs_binaries
@_needs_langs("eng", "spa")
def test_spanish_data_recovers_more_accented_words_on_the_spanish_scans_than_english_alone():
    with_spa = [_accented_words_recovered(i, "eng+spa") for i in ES_IMAGE_IDS]
    english_only = [_accented_words_recovered(i, "eng") for i in ES_IMAGE_IDS]
    assert sum(n for n, _ in with_spa) > sum(n for n, _ in english_only)
    assert sum(total for _, total in with_spa) == sum(total for _, total in english_only) > 0


@needs_binaries
@_needs_langs("eng", "chi_sim")
def test_chinese_simplified_data_reads_a_generated_simplified_image():
    path = os.path.join(FIXTURES_OCR, "zh-sim.png")
    text = "".join(read_document(path, ocr_langs="eng+chi_sim").text.split())
    assert "发票" in text and "总计" in text
    assert "发票" not in "".join(read_document(path, ocr_langs="eng").text.split())


@needs_binaries
@_needs_langs("eng", "chi_tra")
def test_chinese_traditional_data_reads_a_generated_traditional_image():
    path = os.path.join(FIXTURES_OCR, "zh-tra.png")
    out = read_document(path, ocr_langs="eng+chi_tra")
    text = "".join(out.text.split())
    assert out.ocr_lang == "eng+chi_tra"
    assert "發票" in text and "總計" in text

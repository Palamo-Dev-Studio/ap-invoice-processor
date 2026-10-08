# ABOUTME: Document-reading layer: turns an invoice PDF or image into plain text via pdftotext or Tesseract.
# ABOUTME: Shells out to local binaries only (no network); raises ReaderError on any tool failure.
import os
import subprocess
import tempfile
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel

SUBPROCESS_TIMEOUT_S = 60
# A born-digital PDF with fewer non-whitespace characters than this is treated as image-only.
MIN_PDF_TEXT_CHARS = 20
# Tesseract language data used for OCR. Scans default to English plus Spanish; the two Chinese models are opt-in
# (they slow recognition and are not needed for the current corpus). AP_OCR_LANGS overrides the default, for example
# "eng+spa+chi_sim". Codes are checked against this closed set because they reach the tesseract command line.
SUPPORTED_OCR_LANGS = ("eng", "spa", "chi_sim", "chi_tra")
DEFAULT_OCR_LANGS = ("eng", "spa")
OCR_LANGS_ENV = "AP_OCR_LANGS"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


class ReaderError(RuntimeError):
    """Raised when a document cannot be read: missing binary, timeout, nonzero exit, unsupported type."""


class ReaderOutput(BaseModel):
    doc_id: str
    text: str
    method: str  # "pdftotext" | "tesseract"
    ocr_lang: Optional[str] = None  # set only when OCR was used


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run one external tool with a timeout, converting every failure into ReaderError."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderError(f"{cmd[0]} timed out after {SUBPROCESS_TIMEOUT_S}s") from exc
    except FileNotFoundError as exc:
        raise ReaderError(f"{cmd[0]} is not installed or not on PATH") from exc
    if result.returncode != 0:
        raise ReaderError(f"{cmd[0]} exited with code {result.returncode}: {result.stderr.strip()}")
    return result


def resolve_ocr_langs(langs: Union[None, str, Sequence[str]] = None) -> str:
    """Return the tesseract `-l` value ("eng+spa") for `langs`, else for AP_OCR_LANGS, else for the default.

    Accepts a "+" or "," separated string or a sequence of codes; order is kept and duplicates are dropped.
    Raises ReaderError for an empty list or any code outside SUPPORTED_OCR_LANGS.
    """
    if langs is None:
        langs = os.environ.get(OCR_LANGS_ENV) or DEFAULT_OCR_LANGS
    if isinstance(langs, str):
        langs = langs.replace(",", "+").split("+")
    codes: List[str] = []
    for code in (str(c).strip() for c in langs):
        if code not in SUPPORTED_OCR_LANGS:
            raise ReaderError(f"unsupported OCR language {code!r}; supported: {', '.join(SUPPORTED_OCR_LANGS)}")
        if code not in codes:
            codes.append(code)
    if not codes:
        raise ReaderError(f"no OCR language given; supported: {', '.join(SUPPORTED_OCR_LANGS)}")
    return "+".join(codes)


def _ocr_image(image_path: str, lang: str) -> str:
    """OCR one image twice with the `lang` model(s) and join the outputs.

    Tesseract's default binarisation drops the tinted sidebar of the twocol layout (invoice number,
    dates, PO, bill-to), while Sauvola local thresholding (thresholding_method=2) recovers it but loses
    some vendor names. The two passes are complementary, so both outputs are kept.
    """
    default = _run(["tesseract", image_path, "-", "-l", lang]).stdout
    sauvola = _run(["tesseract", image_path, "-", "-l", lang, "-c", "thresholding_method=2"]).stdout
    return default + "\n" + sauvola


def _pdf_text(pdf_path: str) -> str:
    return _run(["pdftotext", "-layout", pdf_path, "-"]).stdout


def _ocr_pdf(pdf_path: str, lang: str) -> str:
    """Rasterise every page of the PDF at 150 dpi and OCR each one."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "page")
        _run(["pdftoppm", "-r", "150", "-png", pdf_path, prefix])
        pages = sorted(name for name in os.listdir(tmp) if name.endswith(".png"))
        if not pages:
            raise ReaderError(f"pdftoppm produced no pages for {pdf_path}")
        return "\n".join(_ocr_image(os.path.join(tmp, name), lang) for name in pages)


def read_document(path: str, ocr_langs: Union[None, str, Sequence[str]] = None) -> ReaderOutput:
    """Read an invoice document (PDF or image) and return its text plus how it was obtained.

    `ocr_langs` picks the Tesseract language data for OCR (see resolve_ocr_langs); it is ignored for a PDF with a
    text layer.
    """
    lang = resolve_ocr_langs(ocr_langs)
    if not os.path.isfile(path):
        raise ReaderError(f"document not found: {path}")
    # An absolute path cannot start with "-", so a file named like an option is never parsed as one by the tools.
    path = os.path.abspath(path)
    doc_id, suffix = os.path.splitext(os.path.basename(path))
    suffix = suffix.lower()

    if suffix == ".pdf":
        text = _pdf_text(path)
        if len("".join(text.split())) >= MIN_PDF_TEXT_CHARS:
            return ReaderOutput(doc_id=doc_id, text=text, method="pdftotext")
        return ReaderOutput(doc_id=doc_id, text=_ocr_pdf(path, lang), method="tesseract", ocr_lang=lang)
    if suffix in IMAGE_SUFFIXES:
        return ReaderOutput(doc_id=doc_id, text=_ocr_image(path, lang), method="tesseract", ocr_lang=lang)
    raise ReaderError(f"unsupported document type {suffix!r}: {path}")

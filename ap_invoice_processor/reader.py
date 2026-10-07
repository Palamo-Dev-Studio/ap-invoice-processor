# ABOUTME: Document-reading layer: turns an invoice PDF or image into plain text via pdftotext or Tesseract.
# ABOUTME: Shells out to local binaries only (no network); raises ReaderError on any tool failure.
import os
import subprocess
import tempfile
from typing import List, Optional

from pydantic import BaseModel

SUBPROCESS_TIMEOUT_S = 60
# A born-digital PDF with fewer non-whitespace characters than this is treated as image-only.
MIN_PDF_TEXT_CHARS = 20
# tessdata on this machine ships only English (plus osd), so every OCR run uses English.
OCR_LANG = "eng"
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


def _ocr_image(image_path: str) -> str:
    """OCR one image twice and join the outputs.

    Tesseract's default binarisation drops the tinted sidebar of the twocol layout (invoice number,
    dates, PO, bill-to), while Sauvola local thresholding (thresholding_method=2) recovers it but loses
    some vendor names. The two passes are complementary, so both outputs are kept.
    """
    default = _run(["tesseract", image_path, "-", "-l", OCR_LANG]).stdout
    sauvola = _run(["tesseract", image_path, "-", "-l", OCR_LANG, "-c", "thresholding_method=2"]).stdout
    return default + "\n" + sauvola


def _pdf_text(pdf_path: str) -> str:
    return _run(["pdftotext", "-layout", pdf_path, "-"]).stdout


def _ocr_pdf(pdf_path: str) -> str:
    """Rasterise every page of the PDF at 150 dpi and OCR each one."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "page")
        _run(["pdftoppm", "-r", "150", "-png", pdf_path, prefix])
        pages = sorted(name for name in os.listdir(tmp) if name.endswith(".png"))
        if not pages:
            raise ReaderError(f"pdftoppm produced no pages for {pdf_path}")
        return "\n".join(_ocr_image(os.path.join(tmp, name)) for name in pages)


def read_document(path: str) -> ReaderOutput:
    """Read an invoice document (PDF or image) and return its text plus how it was obtained."""
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
        return ReaderOutput(doc_id=doc_id, text=_ocr_pdf(path), method="tesseract", ocr_lang=OCR_LANG)
    if suffix in IMAGE_SUFFIXES:
        return ReaderOutput(doc_id=doc_id, text=_ocr_image(path), method="tesseract", ocr_lang=OCR_LANG)
    raise ReaderError(f"unsupported document type {suffix!r}: {path}")

# ABOUTME: Document-reading layer: turns an invoice PDF or image into plain text via pdftotext or Tesseract.
# ABOUTME: Shells out to local binaries only (no network); raises ReaderError on any tool failure.
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
from typing import Callable, List, NamedTuple, Optional, Sequence, Union

from pydantic import BaseModel

SUBPROCESS_TIMEOUT_S = 60
# pdfinfo only parses the file structure, so it gets a much shorter timeout than rasterising or OCR.
PDFINFO_TIMEOUT_S = 10
# Scanned PDFs are rasterised at this resolution, unless a page is so large that its longest edge would pass
# MAX_RASTER_EDGE_PX pixels: a PDF can declare a page of any size, and a 200-inch page at 150 dpi is gigabytes.
# That dpi is a prediction from pdfinfo, so pdftoppm is also told to emit at most MAX_RASTER_EDGE_PX square pixels per
# page whatever the PDF claims (see _ocr_pdf), and every reader subprocess runs under a memory cap (see _run).
RASTER_DPI = 150
MAX_RASTER_EDGE_PX = 2500
# Address-space limit applied to every reader subprocess (pdfinfo, pdftotext, pdftoppm, tesseract) on Linux, in bytes.
# AP_SUBPROCESS_MAX_BYTES overrides it. RLIMIT_AS is only enforced on Linux, which is what the Cloud Run container runs;
# elsewhere (macOS rejects the call outright) no limit is set and only the raster bound above protects the host.
SUBPROCESS_MAX_BYTES_ENV = "AP_SUBPROCESS_MAX_BYTES"
DEFAULT_SUBPROCESS_MAX_BYTES = 1024**3
# When set to a positive integer, a PDF with more pages than this is refused and only the first pages are rasterised.
# The dashboard applies a default of its own (web/uploads.py); the CLI and eval paths have no cap unless this is set.
MAX_PDF_PAGES_ENV = "AP_MAX_PDF_PAGES"
# pdfinfo lists the size of every page from -f to -l (and clamps -l to the page count); this reaches any real document.
_PDFINFO_ALL_PAGES = 100000
_PDFINFO_PAGES = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)
# pdfinfo prints a size of 1e6 points or more in exponent notation ("1e+06 x 144 pts").
_PDF_NUMBER = r"[0-9.]+(?:[eE][+-]?[0-9]+)?"
_PDFINFO_PAGE_SIZE = re.compile(rf"^Page\s+\d+\s+size:\s+({_PDF_NUMBER})\s+x\s+({_PDF_NUMBER})\s+pts", re.MULTILINE)
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


class PdfInfo(NamedTuple):
    pages: int  # total page count of the document
    longest_edge_pts: float  # longest page edge, in points (1/72 inch), among the pages listed


def configured_subprocess_max_bytes() -> int:
    """The per-subprocess memory cap from AP_SUBPROCESS_MAX_BYTES, or the default when it is unset or not a positive integer."""
    try:
        value = int(os.environ.get(SUBPROCESS_MAX_BYTES_ENV, ""))
    except ValueError:
        return DEFAULT_SUBPROCESS_MAX_BYTES
    return value if value >= 1 else DEFAULT_SUBPROCESS_MAX_BYTES


def _memory_limit_hook() -> Optional[Callable[[], None]]:
    """A preexec_fn that caps the child's address space, or None where RLIMIT_AS is not enforced (anything but Linux)."""
    if not sys.platform.startswith("linux"):
        return None
    limit = configured_subprocess_max_bytes()  # read here, in the parent: the hook runs between fork and exec

    def apply_limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply_limit


def _run(cmd: List[str], timeout: int = SUBPROCESS_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run one external tool with a timeout and a memory cap, converting every failure into ReaderError."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            preexec_fn=_memory_limit_hook(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderError(f"{cmd[0]} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise ReaderError(f"{cmd[0]} is not installed or not on PATH") from exc
    except subprocess.SubprocessError as exc:  # includes a failed preexec_fn
        raise ReaderError(f"{cmd[0]} could not be started under its limits: {exc}") from exc
    if result.returncode < 0:  # a tool that runs out of address space typically dies on SIGSEGV or SIGABRT
        raise ReaderError(f"{cmd[0]} was killed by signal {-result.returncode} (likely over the memory limit)")
    if result.returncode != 0:
        raise ReaderError(f"{cmd[0]} exited with code {result.returncode}: {result.stderr.strip()}")
    return result


def configured_max_pdf_pages(default: Optional[int] = None) -> Optional[int]:
    """The page cap from AP_MAX_PDF_PAGES, or `default` when it is unset or not a positive integer."""
    try:
        value = int(os.environ.get(MAX_PDF_PAGES_ENV, ""))
    except ValueError:
        return default
    return value if value >= 1 else default


def pdf_info(pdf_path: str, last_page: int) -> PdfInfo:
    """Read the total page count of a PDF and the longest page edge among pages 1..`last_page`, with pdfinfo.

    Only structure is parsed, never page content. Raises ReaderError when pdfinfo fails or its output has no page data.
    """
    out = _run(["pdfinfo", "-f", "1", "-l", str(last_page), pdf_path], timeout=PDFINFO_TIMEOUT_S).stdout
    pages = _PDFINFO_PAGES.search(out)
    sizes = _PDFINFO_PAGE_SIZE.findall(out)
    if pages is None or not sizes:
        raise ReaderError(f"pdfinfo reported no page count or page size for {pdf_path}")
    # One size line per page from 1 to last_page; a count that disagrees means a line was dropped or not understood.
    expected = min(int(pages.group(1)), last_page)
    if len(sizes) != expected:
        raise ReaderError(f"pdfinfo listed {len(sizes)} page sizes but {expected} were expected for {pdf_path}")
    return PdfInfo(pages=int(pages.group(1)), longest_edge_pts=max(max(float(w), float(h)) for w, h in sizes))


def _raster_dpi(longest_edge_pts: float) -> str:
    """The pdftoppm -r value: 150 dpi, lowered only as far as needed to keep the longest page edge within MAX_RASTER_EDGE_PX.

    pdftoppm's own -scale-to option is not used because it also enlarges small pages, which would change the OCR input
    of every ordinary invoice.
    """
    dpi = math.floor(MAX_RASTER_EDGE_PX * 72 / longest_edge_pts * 100) / 100  # rounded down so the edge never passes the limit
    return f"{max(0.01, min(float(RASTER_DPI), dpi)):g}"


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


def _ocr_pdf(pdf_path: str, lang: str, max_pages: Optional[int] = None, info: Optional[PdfInfo] = None) -> str:
    """Rasterise the PDF's pages (the first `max_pages` when a cap is given) at up to 150 dpi and OCR each one.

    `info` is the pdfinfo result covering those pages; it is read here when the caller has not already done so.
    """
    if info is None:
        info = pdf_info(pdf_path, max_pages or _PDFINFO_ALL_PAGES)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "page")
        # -cropbox renders the box pdfinfo measured (pdftoppm defaults to the MediaBox); -x/-y/-W/-H slice every page to
        # at most MAX_RASTER_EDGE_PX square, which bounds the bitmap whatever size the PDF declares.
        edge = str(MAX_RASTER_EDGE_PX)
        cmd = ["pdftoppm", "-r", _raster_dpi(info.longest_edge_pts), "-cropbox"]
        cmd += ["-x", "0", "-y", "0", "-W", edge, "-H", edge, "-png"]
        if max_pages is not None:
            cmd += ["-l", str(max_pages)]
        _run(cmd + [pdf_path, prefix])
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
        max_pages = configured_max_pdf_pages()
        info = None
        if max_pages is not None:
            info = pdf_info(path, max_pages)
            if info.pages > max_pages:
                raise ReaderError(f"the PDF has {info.pages} pages; the limit is {max_pages} ({MAX_PDF_PAGES_ENV})")
        text = _pdf_text(path)
        if len("".join(text.split())) >= MIN_PDF_TEXT_CHARS:
            return ReaderOutput(doc_id=doc_id, text=text, method="pdftotext")
        return ReaderOutput(doc_id=doc_id, text=_ocr_pdf(path, lang, max_pages, info), method="tesseract", ocr_lang=lang)
    if suffix in IMAGE_SUFFIXES:
        return ReaderOutput(doc_id=doc_id, text=_ocr_image(path, lang), method="tesseract", ocr_lang=lang)
    raise ReaderError(f"unsupported document type {suffix!r}: {path}")

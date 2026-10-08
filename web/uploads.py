# ABOUTME: Validation and temporary staging for invoice documents handed to the dashboard (user uploads or corpus samples).
# ABOUTME: A document is size-capped, type-checked by extension and magic bytes, staged in a private temp dir and removed after use.
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ap_invoice_processor.reader import ReaderError, configured_max_pdf_pages, pdf_info

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
# A small file can declare a huge image (a PNG of 40,000 x 40,000 white pixels is a few hundred KB), and OCR decodes it
# to raw pixels. The dimensions are read from the header, so the check costs nothing and decodes nothing.
MAX_IMAGE_PIXELS = 25_000_000
# Page cap for an uploaded PDF unless AP_MAX_PDF_PAGES says otherwise; every page of a scan is rasterised and OCRed twice.
DEFAULT_MAX_PDF_PAGES = 10
# Extension -> the leading bytes a genuine file of that type starts with. The extension picks the reader branch and the
# magic bytes stop a renamed file (an executable called invoice.pdf) from reaching it.
ALLOWED_TYPES: Dict[str, bytes] = {
    ".pdf": b"%PDF-",
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
}
STAGING_PREFIX = "ap-upload-"
_MAX_STEM_CHARS = 64
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_SAMPLE_SUFFIXES = (".pdf", ".png")


class UploadRejected(Exception):
    """A document the dashboard refuses to process; `status_code` is the HTTP status to answer with."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class StagedDocument:
    directory: str  # private temp dir holding exactly one file; remove_staged deletes it
    path: str  # the document inside `directory`
    stem: str  # sanitised file stem; the pipeline uses it as the document id


def safe_stem(filename: Optional[str]) -> str:
    """A file stem made only of letters, digits, dot, underscore and hyphen, starting with a letter or digit."""
    stem = os.path.splitext(os.path.basename((filename or "").replace("\\", "/")))[0]
    stem = _UNSAFE_CHARS.sub("_", stem).lstrip("._-")[:_MAX_STEM_CHARS].rstrip("._-")
    return stem or "upload"


def _png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from the IHDR chunk, which the PNG format requires to come first; None if it is not there."""
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


# JPEG start-of-frame markers: C0-CF except DHT (C4), JPG (C8) and DAC (CC), which share the range but are not frames.
_JPEG_FRAME_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Markers that stand alone, with no length field after them.
_JPEG_BARE_MARKERS = frozenset({0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)})


def _jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from the first start-of-frame segment; None if the segments run out or the scan starts first."""
    i, end = 2, len(data)
    while i + 1 < end:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte before a marker
            i += 1
            continue
        i += 2
        if marker in _JPEG_BARE_MARKERS:
            continue
        if marker == 0xDA or i + 2 > end:  # start of scan: no frame header can follow
            return None
        length = struct.unpack(">H", data[i : i + 2])[0]
        if length < 2:  # a segment always counts its own two length bytes; anything less would stall the walk
            return None
        if marker in _JPEG_FRAME_MARKERS:
            if length < 7 or i + 7 > end:
                return None
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        i += length
    return None


def _check_image_size(extension: str, data: bytes) -> None:
    dimensions = _png_dimensions(data) if extension == ".png" else _jpeg_dimensions(data)
    if dimensions is None or 0 in dimensions:
        raise UploadRejected(415, f"The file content is not a valid {extension.lstrip('.').upper()}.")
    if dimensions[0] * dimensions[1] > MAX_IMAGE_PIXELS:
        raise UploadRejected(413, f"The image is larger than the {MAX_IMAGE_PIXELS // 1_000_000} megapixel limit.")


def check_pdf_page_limit(path: str) -> None:
    """Raise UploadRejected(413) when the PDF at `path` has more pages than AP_MAX_PDF_PAGES (default DEFAULT_MAX_PDF_PAGES).

    Blocking: it runs pdfinfo. A PDF that pdfinfo cannot read passes here, because the reader runs the same tool, fails
    on it the same way and sends the document to human review with the reason.
    """
    cap = configured_max_pdf_pages(DEFAULT_MAX_PDF_PAGES)
    try:
        pages = pdf_info(path, last_page=1).pages
    except ReaderError:
        return
    if pages > cap:
        raise UploadRejected(413, f"The PDF has {pages} pages; the limit is {cap}.")


def validate_document(filename: Optional[str], data: bytes) -> str:
    """Return the lower-case extension of an acceptable document, or raise UploadRejected.

    The checks run in order: not empty, within MAX_UPLOAD_BYTES, an allowed extension, leading bytes that match it, and
    for an image, a readable header whose dimensions stay within MAX_IMAGE_PIXELS. The PDF page cap needs a file on
    disk, so check_pdf_page_limit applies it after staging.
    """
    if not data:
        raise UploadRejected(400, "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(413, f"The file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    extension = os.path.splitext((filename or "").replace("\\", "/"))[1].lower()
    magic = ALLOWED_TYPES.get(extension)
    if magic is None:
        raise UploadRejected(415, "Only PDF, PNG and JPEG files are accepted.")
    if not data.startswith(magic):
        raise UploadRejected(415, f"The file content is not a valid {extension.lstrip('.').upper()}.")
    if extension != ".pdf":
        _check_image_size(extension, data)
    return extension


def stage_document(filename: Optional[str], data: bytes) -> StagedDocument:
    """Validate `data` and write it to a new private temp dir; the caller must call remove_staged on the result."""
    extension = validate_document(filename, data)
    stem = safe_stem(filename)
    directory = tempfile.mkdtemp(prefix=STAGING_PREFIX)  # mkdtemp creates the dir owner-only (0700)
    try:
        path = os.path.join(directory, stem + extension)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        remove_staged(directory)
        raise
    return StagedDocument(directory=directory, path=path, stem=stem)


def remove_staged(directory: Optional[str]) -> None:
    """Delete a staging directory and everything in it; a directory that is already gone is not an error."""
    if directory:
        shutil.rmtree(directory, ignore_errors=True)


def list_samples(corpus_dir: str) -> List[Dict[str, str]]:
    """The corpus documents a visitor may pick: [{"id", "label", "kind"}], PDFs then scans/photos, each sorted by id."""
    samples = []
    for subdir, kind in (("pdf", "PDF"), ("images", "scan/photo")):
        folder = os.path.join(corpus_dir, subdir)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            stem, extension = os.path.splitext(name)
            if extension.lower() in _SAMPLE_SUFFIXES:
                samples.append({"id": name, "label": f"{stem} ({kind})", "kind": kind})
    return samples


def stage_sample(corpus_dir: str, sample_id: str) -> StagedDocument:
    """Copy one listed corpus document into a staging dir, so the run can delete its copy and never the corpus file.

    `sample_id` must equal an id from list_samples; anything else (including a path) raises UploadRejected(404).
    """
    for sample in list_samples(corpus_dir):
        if sample["id"] == sample_id:
            subdir = "pdf" if sample["kind"] == "PDF" else "images"
            with open(os.path.join(corpus_dir, subdir, sample_id), "rb") as f:
                data = f.read()
            return stage_document(sample_id, data)
    raise UploadRejected(404, "Unknown sample document.")

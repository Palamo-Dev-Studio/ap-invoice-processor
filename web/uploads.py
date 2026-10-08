# ABOUTME: Validation and temporary staging for invoice documents handed to the dashboard (user uploads or corpus samples).
# ABOUTME: A document is size-capped, type-checked by extension and magic bytes, staged in a private temp dir and removed after use.
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
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


def validate_document(filename: Optional[str], data: bytes) -> str:
    """Return the lower-case extension of an acceptable document, or raise UploadRejected.

    The checks run in order: not empty, within MAX_UPLOAD_BYTES, an allowed extension, and leading bytes that match it.
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

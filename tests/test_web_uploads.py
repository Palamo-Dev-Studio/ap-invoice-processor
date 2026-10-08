# ABOUTME: Unit tests for web/uploads.py: size, type and magic-byte validation, filename sanitising, staging and cleanup.
# ABOUTME: Also covers the corpus sample list and the guarantee that a sample id can never name a path.
import os
import stat

import pytest

from ap_invoice_processor import reader
from document_builders import jpeg_header, png_header
from web import uploads
from web.uploads import MAX_UPLOAD_BYTES, UploadRejected

PDF_BYTES = b"%PDF-1.4\n" + b"x" * 64
PNG_BYTES = png_header(100, 100)
JPEG_BYTES = jpeg_header(100, 100)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")


@pytest.fixture(autouse=True)
def private_tmp(tmp_path, monkeypatch):
    """Point tempfile at a per-test directory so a leaked staging dir is visible."""
    staging_root = tmp_path / "tmp"
    staging_root.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(staging_root))
    return staging_root


# --- validate_document -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,data,extension",
    [
        ("invoice.pdf", PDF_BYTES, ".pdf"),
        ("INVOICE.PDF", PDF_BYTES, ".pdf"),
        ("scan.png", PNG_BYTES, ".png"),
        ("photo.jpg", JPEG_BYTES, ".jpg"),
        ("photo.JPEG", JPEG_BYTES, ".jpeg"),
    ],
)
def test_accepts_each_allowed_type(filename, data, extension):
    assert uploads.validate_document(filename, data) == extension


def test_a_file_of_exactly_the_size_limit_is_accepted_and_one_byte_more_is_not():
    at_limit = PDF_BYTES + b"0" * (MAX_UPLOAD_BYTES - len(PDF_BYTES))
    assert len(at_limit) == MAX_UPLOAD_BYTES
    assert uploads.validate_document("a.pdf", at_limit) == ".pdf"
    with pytest.raises(UploadRejected) as err:
        uploads.validate_document("a.pdf", at_limit + b"0")
    assert err.value.status_code == 413


def test_the_size_limit_is_five_megabytes():
    assert MAX_UPLOAD_BYTES == 5 * 1024 * 1024


@pytest.mark.parametrize("filename", ["notes.txt", "run.exe", "page.html", "archive.zip", "noextension", "", None, "a.pdf.exe", ".pdf"])
def test_rejects_other_extensions(filename):
    with pytest.raises(UploadRejected) as err:
        uploads.validate_document(filename, PDF_BYTES)
    assert err.value.status_code == 415


@pytest.mark.parametrize(
    "filename,data",
    [
        ("invoice.pdf", b"MZ\x90\x00 an executable renamed to pdf"),
        ("invoice.pdf", PNG_BYTES),
        ("scan.png", PDF_BYTES),
        ("photo.jpg", PNG_BYTES),
        ("scan.png", b"<html>not a png</html>"),
    ],
)
def test_rejects_content_that_does_not_match_the_extension(filename, data):
    with pytest.raises(UploadRejected) as err:
        uploads.validate_document(filename, data)
    assert err.value.status_code == 415


def test_rejects_an_empty_file():
    with pytest.raises(UploadRejected) as err:
        uploads.validate_document("a.pdf", b"")
    assert err.value.status_code == 400


# --- safe_stem ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("en-001.pdf", "en-001"),
        ("../../etc/passwd.pdf", "passwd"),
        ("..\\..\\windows\\invoice.pdf", "invoice"),
        ("My Invoice (final) #3.pdf", "My_Invoice_final_3"),
        ("<script>alert(1)</script>.png", "script"),
        ("<img onerror=x>.png", "img_onerror_x"),
        ("   .pdf", "upload"),
        (None, "upload"),
        ("na\x00me.pdf", "na_me"),
    ],
)
def test_safe_stem_keeps_only_a_safe_alphabet(filename, expected):
    assert uploads.safe_stem(filename) == expected


def test_safe_stem_is_length_limited():
    assert len(uploads.safe_stem("a" * 500 + ".pdf")) <= 64


# --- staging and cleanup -----------------------------------------------------------------------------------------------


def test_staging_writes_one_private_file_in_its_own_directory(private_tmp):
    staged = uploads.stage_document("My Invoice.pdf", PDF_BYTES)
    assert os.path.dirname(staged.path) == staged.directory
    assert os.path.dirname(staged.directory) == str(private_tmp)
    assert os.listdir(staged.directory) == ["My_Invoice.pdf"]
    assert staged.stem == "My_Invoice"
    with open(staged.path, "rb") as f:
        assert f.read() == PDF_BYTES
    assert stat.S_IMODE(os.stat(staged.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(staged.directory).st_mode) == 0o700


def test_a_traversal_filename_cannot_escape_the_staging_directory(private_tmp):
    staged = uploads.stage_document("../../outside.pdf", PDF_BYTES)
    assert os.path.dirname(staged.path) == staged.directory
    assert not (private_tmp.parent / "outside.pdf").exists()
    assert not (private_tmp / "outside.pdf").exists()


def test_remove_staged_deletes_the_directory_and_tolerates_a_second_call(private_tmp):
    staged = uploads.stage_document("a.pdf", PDF_BYTES)
    uploads.remove_staged(staged.directory)
    assert not os.path.exists(staged.directory)
    uploads.remove_staged(staged.directory)
    uploads.remove_staged(None)
    assert os.listdir(private_tmp) == []


def test_a_rejected_document_leaves_nothing_behind(private_tmp):
    for filename, data in (("a.txt", b"hello"), ("a.pdf", PNG_BYTES), ("a.pdf", b""), ("a.pdf", PDF_BYTES * (MAX_UPLOAD_BYTES // 8))):
        with pytest.raises(UploadRejected):
            uploads.stage_document(filename, data)
    assert os.listdir(private_tmp) == []


def test_a_failed_write_removes_the_staging_directory(private_tmp, monkeypatch):
    def boom(fd, *args, **kwargs):
        os.close(fd)
        raise OSError("disk full")

    monkeypatch.setattr(uploads.os, "fdopen", boom)
    with pytest.raises(OSError, match="disk full"):
        uploads.stage_document("a.pdf", PDF_BYTES)
    assert os.listdir(private_tmp) == []


# --- corpus samples ----------------------------------------------------------------------------------------------------


def test_the_sample_list_covers_the_pdfs_and_the_scans():
    samples = uploads.list_samples(CORPUS)
    ids = [s["id"] for s in samples]
    assert "en-001.pdf" in ids and "es-012.pdf" in ids and "en-004-scan.png" in ids
    assert len(ids) == 40
    assert len(set(ids)) == len(ids)


def test_the_sample_list_is_empty_when_the_corpus_is_absent(tmp_path):
    assert uploads.list_samples(str(tmp_path / "missing")) == []


def test_staging_a_sample_copies_it_and_leaves_the_corpus_file_alone(private_tmp):
    staged = uploads.stage_sample(CORPUS, "en-001.pdf")
    assert staged.stem == "en-001"
    assert staged.path != os.path.join(CORPUS, "pdf", "en-001.pdf")
    assert open(staged.path, "rb").read(5) == b"%PDF-"
    uploads.remove_staged(staged.directory)
    assert os.path.isfile(os.path.join(CORPUS, "pdf", "en-001.pdf"))


def test_a_scan_sample_keeps_its_name_so_its_fixture_is_found(private_tmp):
    staged = uploads.stage_sample(CORPUS, "en-004-scan.png")
    assert staged.stem == "en-004-scan"
    assert staged.path.endswith("en-004-scan.png")
    uploads.remove_staged(staged.directory)


@pytest.mark.parametrize("sample_id", ["../../../etc/passwd", "pdf/en-001.pdf", "/etc/passwd", "en-001", "EN-001.PDF", "", "nope.pdf", "../README.md"])
def test_a_sample_id_that_is_not_listed_is_refused(sample_id, private_tmp):
    with pytest.raises(UploadRejected) as err:
        uploads.stage_sample(CORPUS, sample_id)
    assert err.value.status_code == 404
    assert os.listdir(private_tmp) == []


# --- image pixel ceiling -----------------------------------------------------------------------------------------------


def test_the_pixel_ceiling_is_twenty_five_megapixels():
    assert uploads.MAX_IMAGE_PIXELS == 25_000_000


@pytest.mark.parametrize(
    "filename,build",
    [
        ("a.png", png_header),
        ("a.jpg", jpeg_header),
        ("a.jpeg", lambda w, h: jpeg_header(w, h, progressive=True)),
        ("a.jpg", lambda w, h: jpeg_header(w, h, leading_segments=6)),
    ],
)
def test_an_image_of_exactly_twenty_five_megapixels_is_accepted_and_one_more_pixel_is_not(filename, build):
    assert uploads.validate_document(filename, build(5000, 5000))
    assert uploads.validate_document(filename, build(6250, 4000))
    # A JPEG dimension is 16 bits, so its extreme cases are 65535 pixels along one edge.
    extremes = ((25_000_001, 1), (1, 25_000_001)) if filename.endswith("png") else ((65535, 382), (382, 65535))
    for width, height in extremes + ((5001, 5000), (5000, 5001)):
        with pytest.raises(UploadRejected) as err:
            uploads.validate_document(filename, build(width, height))
        assert err.value.status_code == 413
        assert "25" in err.value.detail and "megapixel" in err.value.detail


def test_one_long_edge_alone_is_not_enough_to_trip_the_ceiling():
    assert uploads.validate_document("a.png", png_header(20_000, 1_000))


@pytest.mark.parametrize(
    "filename,data",
    [
        ("a.png", b"\x89PNG\r\n\x1a\n" + b"x" * 64),  # signature but no IHDR chunk
        ("a.png", b"\x89PNG\r\n\x1a\n"),  # truncated
        ("a.png", png_header(0, 100)),  # a zero dimension is not an image
        ("a.jpg", b"\xff\xd8\xff\xe0" + b"x" * 64),  # no frame header
        ("a.jpg", b"\xff\xd8\xff"),  # truncated
        ("a.jpg", jpeg_header(100, 100)[:12]),  # cut off inside the first segment
    ],
)
def test_an_image_whose_size_cannot_be_read_is_refused_rather_than_trusted(filename, data):
    with pytest.raises(UploadRejected) as err:
        uploads.validate_document(filename, data)
    assert err.value.status_code == 415


def test_an_oversized_image_leaves_nothing_on_disk(private_tmp):
    with pytest.raises(UploadRejected):
        uploads.stage_document("huge.png", png_header(10_000, 10_000))
    assert os.listdir(private_tmp) == []


# --- PDF page cap -----------------------------------------------------------------------------------------------------


def _pages(monkeypatch, count):
    monkeypatch.setattr(uploads, "pdf_info", lambda path, last_page: reader.PdfInfo(pages=count, longest_edge_pts=792))


def test_the_default_page_cap_is_ten(monkeypatch):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    assert uploads.DEFAULT_MAX_PDF_PAGES == 10
    _pages(monkeypatch, 10)
    uploads.check_pdf_page_limit("x.pdf")
    _pages(monkeypatch, 11)
    with pytest.raises(UploadRejected) as err:
        uploads.check_pdf_page_limit("x.pdf")
    assert err.value.status_code == 413
    assert "11 pages" in err.value.detail and "10" in err.value.detail


def test_the_page_cap_follows_ap_max_pdf_pages(monkeypatch):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "3")
    _pages(monkeypatch, 3)
    uploads.check_pdf_page_limit("x.pdf")
    _pages(monkeypatch, 4)
    with pytest.raises(UploadRejected, match="limit is 3"):
        uploads.check_pdf_page_limit("x.pdf")
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "40")
    _pages(monkeypatch, 40)
    uploads.check_pdf_page_limit("x.pdf")


def test_a_pdf_pdfinfo_cannot_read_is_left_to_the_reader_which_refuses_it(monkeypatch):
    def unreadable(path, last_page):
        raise reader.ReaderError("pdfinfo exited with code 1")

    monkeypatch.setattr(uploads, "pdf_info", unreadable)
    uploads.check_pdf_page_limit("x.pdf")

# ABOUTME: Tests for the reader's resource limits: the page cap, the pdftoppm page range, the raster ceiling and the subprocess memory cap.
# ABOUTME: Unit tests fake the subprocess layer; tests marked needs_poppler also drive the real pdfinfo and pdftoppm.
import shutil
import struct
import subprocess
import sys

import pytest

from ap_invoice_processor import reader
from ap_invoice_processor.reader import ReaderError, read_document
from document_builders import make_pdf

needs_poppler = pytest.mark.skipif(
    not (shutil.which("pdfinfo") and shutil.which("pdftoppm")), reason="poppler-utils not installed"
)


def _pdfinfo_output(pages, sizes):
    lines = ["Title:           x", f"Pages:           {pages}", "Encrypted:       no"]
    for number, (w, h) in enumerate(sizes, start=1):
        lines.append(f"Page {number:4d} size:  {w} x {h} pts")
        lines.append(f"Page {number:4d} rot:   0")
    return "\n".join(lines) + "\n"


class FakeTools:
    """Replaces reader.subprocess.run; records every argv and answers pdfinfo, pdftotext, pdftoppm and tesseract."""

    def __init__(self, monkeypatch, pages=1, sizes=None, pdf_text="", pdfinfo_returncode=0):
        self.calls = []
        self.kwargs = []
        self.pages = pages
        self.sizes = sizes if sizes is not None else [(612, 792)] * pages
        self.pdf_text = pdf_text
        self.pdfinfo_returncode = pdfinfo_returncode
        monkeypatch.setattr(reader.subprocess, "run", self.run)

    def run(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.kwargs.append(dict(kwargs))
        if cmd[0] == "pdfinfo":
            if self.pdfinfo_returncode:
                return subprocess.CompletedProcess(cmd, self.pdfinfo_returncode, stdout="", stderr="Syntax Error")
            if "-l" in cmd:  # pdfinfo lists the sizes of pages -f..-l only, and clamps -l to the page count
                shown = self.sizes[: int(cmd[cmd.index("-l") + 1])]
            else:
                shown = self.sizes[:1]
            return subprocess.CompletedProcess(cmd, 0, stdout=_pdfinfo_output(self.pages, shown), stderr="")
        if cmd[0] == "pdftotext":
            return subprocess.CompletedProcess(cmd, 0, stdout=self.pdf_text, stderr="")
        if cmd[0] == "pdftoppm":
            open(cmd[-1] + "-1.png", "wb").close()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def names(self):
        return [c[0] for c in self.calls]

    def first(self, tool):
        return next(c for c in self.calls if c[0] == tool)

    def kwargs_of(self, tool):
        return [kw for call, kw in zip(self.calls, self.kwargs) if call[0] == tool]


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return str(path)


# --- configured_max_pdf_pages -------------------------------------------------------------------------------------------


def test_the_cap_comes_from_the_environment_and_otherwise_the_default(monkeypatch):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    assert reader.configured_max_pdf_pages() is None
    assert reader.configured_max_pdf_pages(10) == 10
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "4")
    assert reader.configured_max_pdf_pages() == 4
    assert reader.configured_max_pdf_pages(10) == 4


@pytest.mark.parametrize("value", ["", "ten", "0", "-3", "2.5"])
def test_an_unusable_cap_value_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, value)
    assert reader.configured_max_pdf_pages() is None
    assert reader.configured_max_pdf_pages(10) == 10


# --- pdf_info ---------------------------------------------------------------------------------------------------------


def test_pdf_info_reports_the_page_count_and_the_longest_page_edge(monkeypatch, pdf_file):
    tools = FakeTools(monkeypatch, pages=3, sizes=[(612, 792), (595.5, 842), (300, 1200)])
    info = reader.pdf_info(pdf_file, last_page=3)
    assert info.pages == 3
    assert info.longest_edge_pts == 1200
    argv = tools.first("pdfinfo")
    assert argv[argv.index("-f") + 1] == "1" and argv[argv.index("-l") + 1] == "3"


def test_pdf_info_reads_the_total_page_count_even_when_only_the_first_page_size_is_asked_for(monkeypatch, pdf_file):
    FakeTools(monkeypatch, pages=500, sizes=[(612, 792)] * 500)
    info = reader.pdf_info(pdf_file, last_page=1)
    assert info.pages == 500 and info.longest_edge_pts == 792


def test_pdf_info_runs_under_a_short_timeout(monkeypatch, pdf_file):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(cmd, 0, stdout=_pdfinfo_output(1, [(612, 792)]), stderr="")

    monkeypatch.setattr(reader.subprocess, "run", fake_run)
    reader.pdf_info(pdf_file, last_page=1)
    assert seen["timeout"] == reader.PDFINFO_TIMEOUT_S < reader.SUBPROCESS_TIMEOUT_S


def test_pdf_info_turns_a_failed_or_unreadable_pdfinfo_into_a_reader_error(monkeypatch, pdf_file):
    FakeTools(monkeypatch, pdfinfo_returncode=1)
    with pytest.raises(ReaderError, match="pdfinfo"):
        reader.pdf_info(pdf_file, last_page=1)

    def garbage(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="nothing useful\n", stderr="")

    monkeypatch.setattr(reader.subprocess, "run", garbage)
    with pytest.raises(ReaderError, match="page"):
        reader.pdf_info(pdf_file, last_page=1)


def _fixed_pdfinfo(monkeypatch, stdout):
    monkeypatch.setattr(
        reader.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    )


def test_pdf_info_reads_page_sizes_that_pdfinfo_prints_in_exponent_notation(monkeypatch, pdf_file):
    # pdfinfo prints a size of 1e6 points or more as "1e+06"; a regex that only knows plain decimals drops that page.
    FakeTools(monkeypatch, pages=3, sizes=[(612, 792), ("1e+06", 144), ("1.5E+07", 99)])
    assert reader.pdf_info(pdf_file, last_page=3).longest_edge_pts == 1.5e7


def test_pdf_info_refuses_output_that_lists_fewer_page_sizes_than_pages_asked_for(monkeypatch, pdf_file):
    _fixed_pdfinfo(monkeypatch, _pdfinfo_output(3, [(612, 792), (612, 792)]))  # 3 pages asked for, 2 listed
    with pytest.raises(ReaderError, match="2 page sizes.*3"):
        reader.pdf_info(pdf_file, last_page=3)


def test_pdf_info_expects_one_size_line_per_page_up_to_the_last_page_asked_for(monkeypatch, pdf_file):
    _fixed_pdfinfo(monkeypatch, _pdfinfo_output(5, [(612, 792)] * 2))  # asked for pages 1-2 of 5: two lines is right
    assert reader.pdf_info(pdf_file, last_page=2).pages == 5
    _fixed_pdfinfo(monkeypatch, _pdfinfo_output(1, [(612, 792)] * 2))  # a 1-page file listing 2 sizes is not
    with pytest.raises(ReaderError, match="2 page sizes.*1"):
        reader.pdf_info(pdf_file, last_page=3)


# --- page cap in read_document ----------------------------------------------------------------------------------------


def test_a_pdf_over_the_cap_is_refused_before_any_text_or_raster_tool_runs(monkeypatch, pdf_file):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "10")
    tools = FakeTools(monkeypatch, pages=11, pdf_text="plenty of invoice text here")
    with pytest.raises(ReaderError, match="11 pages.*10"):
        read_document(pdf_file)
    assert tools.names() == ["pdfinfo"]


def test_a_pdf_of_exactly_the_cap_is_read(monkeypatch, pdf_file):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "10")
    FakeTools(monkeypatch, pages=10, pdf_text="plenty of invoice text here")
    assert read_document(pdf_file).method == "pdftotext"


def test_without_a_cap_the_reader_never_calls_pdfinfo_for_a_text_pdf(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    tools = FakeTools(monkeypatch, pages=500, pdf_text="plenty of invoice text here")
    assert read_document(pdf_file).method == "pdftotext"
    assert "pdfinfo" not in tools.names()


def test_pdftoppm_is_told_the_last_page_when_a_cap_is_set_and_not_otherwise(monkeypatch, pdf_file):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "3")
    tools = FakeTools(monkeypatch, pages=2)
    read_document(pdf_file)
    argv = tools.first("pdftoppm")
    assert argv[argv.index("-l") + 1] == "3"

    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV)
    tools = FakeTools(monkeypatch, pages=2)
    read_document(pdf_file)
    assert "-l" not in tools.first("pdftoppm")


def test_a_pdf_with_fewer_pages_than_the_cap_asks_pdfinfo_for_no_more_than_the_cap(monkeypatch, pdf_file):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "3")
    tools = FakeTools(monkeypatch, pages=2)
    read_document(pdf_file)
    argv = tools.first("pdfinfo")
    assert argv[argv.index("-l") + 1] == "3"


# --- raster ceiling ---------------------------------------------------------------------------------------------------


def _pdftoppm_dpi(tools):
    argv = tools.first("pdftoppm")
    return float(argv[argv.index("-r") + 1])


def test_a_normal_page_is_still_rendered_at_150_dpi(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    for size in ((612, 792), (595, 842), (842, 1190)):  # letter, A4, A3
        tools = FakeTools(monkeypatch, sizes=[size])
        read_document(pdf_file)
        assert _pdftoppm_dpi(tools) == reader.RASTER_DPI == 150


def test_a_giant_declared_page_is_rendered_no_larger_than_the_pixel_ceiling(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    tools = FakeTools(monkeypatch, sizes=[(14400, 14400)])  # 200 inch square
    read_document(pdf_file)
    edge_px = 14400 / 72 * _pdftoppm_dpi(tools)
    assert edge_px <= reader.MAX_RASTER_EDGE_PX == 2500
    assert edge_px > 2400, "the clamp should use the ceiling, not shrink the page far below it"


def test_the_largest_page_of_a_mixed_pdf_sets_the_resolution(monkeypatch, pdf_file):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "5")
    tools = FakeTools(monkeypatch, pages=3, sizes=[(612, 792), (612, 792), (7200, 3600)])
    read_document(pdf_file)
    assert 7200 / 72 * _pdftoppm_dpi(tools) <= 2500


def test_pdftoppm_renders_the_crop_box_pdfinfo_measured_and_never_more_than_the_ceiling_per_page(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    tools = FakeTools(monkeypatch)
    read_document(pdf_file)
    argv = tools.first("pdftoppm")
    assert "-cropbox" in argv
    for flag, value in (("-x", "0"), ("-y", "0"), ("-W", "2500"), ("-H", "2500")):
        assert argv[argv.index(flag) + 1] == value
    assert reader.MAX_RASTER_EDGE_PX == 2500


def test_a_page_size_in_exponent_notation_still_lowers_the_resolution(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    tools = FakeTools(monkeypatch, sizes=[("1e+06", 144)])
    read_document(pdf_file)
    assert 1e6 / 72 * _pdftoppm_dpi(tools) <= 2500


def test_an_unreadable_page_size_stops_the_rasterisation(monkeypatch, pdf_file):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    tools = FakeTools(monkeypatch, pdfinfo_returncode=1)
    with pytest.raises(ReaderError, match="pdfinfo"):
        read_document(pdf_file)
    assert "pdftoppm" not in tools.names()


# --- the real poppler tools ---------------------------------------------------------------------------------------------


@needs_poppler
def test_pdf_info_against_the_real_pdfinfo(tmp_path):
    path = tmp_path / "three.pdf"
    path.write_bytes(make_pdf(pages=3, width_pts=300, height_pts=900))
    info = reader.pdf_info(str(path), last_page=3)
    assert info.pages == 3 and info.longest_edge_pts == 900


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def _rasterised_sizes(monkeypatch, tmp_path, pdf_bytes):
    """Run the real pdfinfo and pdftoppm through read_document, with OCR replaced by a recorder of each page image's size."""
    path = tmp_path / "scan.pdf"
    path.write_bytes(pdf_bytes)
    sizes = []

    def record(image_path, lang):
        sizes.append(_png_size(image_path))
        return ""

    monkeypatch.setattr(reader, "_ocr_image", record)
    monkeypatch.setattr(reader, "_pdf_text", lambda p: "")
    read_document(str(path))
    return sizes


@needs_poppler
def test_the_real_rasteriser_keeps_a_giant_page_inside_the_ceiling(monkeypatch, tmp_path):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    sizes = _rasterised_sizes(monkeypatch, tmp_path, make_pdf(pages=1, width_pts=14400, height_pts=14400))
    assert sizes and max(max(s) for s in sizes) <= 2500


@needs_poppler
def test_the_real_rasteriser_keeps_a_letter_page_at_150_dpi(monkeypatch, tmp_path):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    assert _rasterised_sizes(monkeypatch, tmp_path, make_pdf(pages=1)) == [(1275, 1650)]


@needs_poppler
def test_the_real_rasteriser_stops_at_the_cap(monkeypatch, tmp_path):
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "3")
    assert len(_rasterised_sizes(monkeypatch, tmp_path, make_pdf(pages=3))) == 3
    with pytest.raises(ReaderError, match="4 pages"):
        _rasterised_sizes(monkeypatch, tmp_path, make_pdf(pages=4))


@needs_poppler
def test_the_real_rasteriser_renders_the_crop_box_not_the_larger_media_box(monkeypatch, tmp_path):
    # pdfinfo measures the CropBox (letter, so 150 dpi) while pdftoppm renders the MediaBox unless told to use the CropBox.
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    pdf = make_pdf(width_pts=14400, height_pts=14400, crop_box=(612, 792))
    assert _rasterised_sizes(monkeypatch, tmp_path, pdf) == [(1275, 1650)]


@needs_poppler
def test_the_real_pdfinfo_and_rasteriser_handle_a_page_size_pdfinfo_prints_in_exponent_notation(monkeypatch, tmp_path):
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    path = tmp_path / "wide.pdf"
    path.write_bytes(make_pdf(page_sizes=[(612, 792), (1000000, 144)]))
    assert reader.pdf_info(str(path), last_page=2).longest_edge_pts == 1e6
    sizes = _rasterised_sizes(monkeypatch, tmp_path, path.read_bytes())
    assert len(sizes) == 2 and max(max(s) for s in sizes) <= 2500


@needs_poppler
def test_the_rasteriser_bounds_every_page_even_when_the_predicted_resolution_is_wrong(monkeypatch, tmp_path):
    # Nothing here trusts the pdfinfo prediction: it is forced to 150 dpi for a 5000-point page, which would be 10417 px.
    monkeypatch.delenv(reader.MAX_PDF_PAGES_ENV, raising=False)
    monkeypatch.setattr(reader, "_raster_dpi", lambda longest_edge_pts: "150")
    sizes = _rasterised_sizes(monkeypatch, tmp_path, make_pdf(width_pts=5000, height_pts=5000))
    assert sizes == [(2500, 2500)]


# --- the memory cap on every reader subprocess -------------------------------------------------------------------------


class RlimitRecorder:
    """Stands in for reader.resource.setrlimit and records each call."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(reader.resource, "setrlimit", lambda which, limits: self.calls.append((which, limits)))


def test_the_memory_cap_comes_from_the_environment_and_otherwise_is_one_gibibyte(monkeypatch):
    monkeypatch.delenv(reader.SUBPROCESS_MAX_BYTES_ENV, raising=False)
    assert reader.configured_subprocess_max_bytes() == 1024**3 == reader.DEFAULT_SUBPROCESS_MAX_BYTES
    monkeypatch.setenv(reader.SUBPROCESS_MAX_BYTES_ENV, "2000000000")
    assert reader.configured_subprocess_max_bytes() == 2_000_000_000


@pytest.mark.parametrize("value", ["", "lots", "0", "-5", "1.5"])
def test_an_unusable_memory_cap_value_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv(reader.SUBPROCESS_MAX_BYTES_ENV, value)
    assert reader.configured_subprocess_max_bytes() == reader.DEFAULT_SUBPROCESS_MAX_BYTES


def test_on_linux_every_reader_tool_runs_with_an_address_space_limit(monkeypatch, pdf_file):
    monkeypatch.setattr(reader.sys, "platform", "linux")
    monkeypatch.setenv(reader.SUBPROCESS_MAX_BYTES_ENV, "123456789")
    monkeypatch.setenv(reader.MAX_PDF_PAGES_ENV, "3")  # makes the reader call pdfinfo as well as pdftoppm and tesseract
    tools = FakeTools(monkeypatch)
    read_document(pdf_file)
    assert {"pdfinfo", "pdftotext", "pdftoppm", "tesseract"} <= set(tools.names())
    rlimit = RlimitRecorder(monkeypatch)
    for tool in ("pdfinfo", "pdftotext", "pdftoppm", "tesseract"):
        for kwargs in tools.kwargs_of(tool):
            hook = kwargs["preexec_fn"]
            assert callable(hook), f"{tool} runs without a memory limit"
            hook()
    assert rlimit.calls and set(rlimit.calls) == {(reader.resource.RLIMIT_AS, (123456789, 123456789))}


def test_the_memory_limit_hook_is_not_installed_off_linux(monkeypatch, pdf_file):
    # macOS rejects RLIMIT_AS outright, so a hook there would make every subprocess fail to start.
    monkeypatch.setattr(reader.sys, "platform", "darwin")
    tools = FakeTools(monkeypatch)
    read_document(pdf_file)
    assert tools.calls and all(kw["preexec_fn"] is None for kw in tools.kwargs)


def test_a_subprocess_that_cannot_be_limited_is_a_reader_error(monkeypatch, pdf_file):
    monkeypatch.setattr(reader.sys, "platform", "linux")

    def refuse(cmd, **kwargs):
        raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

    monkeypatch.setattr(reader.subprocess, "run", refuse)
    with pytest.raises(ReaderError, match="pdftotext.*preexec"):
        reader._pdf_text(pdf_file)


def test_a_tool_killed_by_a_signal_is_a_reader_error_that_names_the_memory_limit(monkeypatch, pdf_file):
    monkeypatch.setattr(
        reader.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, -11, stdout="", stderr="")
    )
    with pytest.raises(ReaderError, match="signal 11.*memory limit"):
        reader._pdf_text(pdf_file)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is only enforced on Linux, the Cloud Run container")
def test_a_real_child_that_allocates_past_the_limit_is_stopped_and_reported(monkeypatch):
    monkeypatch.setenv(reader.SUBPROCESS_MAX_BYTES_ENV, str(512 * 1024**2))
    fits = reader._run([sys.executable, "-c", "x = bytearray(64 * 1024 * 1024)"])
    assert fits.returncode == 0
    with pytest.raises(ReaderError):
        reader._run([sys.executable, "-c", "x = bytearray(1024 * 1024 * 1024)"])

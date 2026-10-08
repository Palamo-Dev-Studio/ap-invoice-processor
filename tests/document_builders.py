# ABOUTME: Test helpers that build tiny well-formed PDF, PNG and JPEG byte strings with chosen page counts and dimensions.
# ABOUTME: The PDFs are blank valid files for poppler; the PNG/JPEG outputs are header-only, enough for dimension checks.
import struct
from typing import Optional, Sequence, Tuple


def make_pdf(
    pages: int = 1,
    width_pts: float = 612,
    height_pts: float = 792,
    crop_box: Optional[Tuple[float, float]] = None,
    page_sizes: Optional[Sequence[Tuple[float, float]]] = None,
) -> bytes:
    """A valid blank PDF with `pages` pages, each `width_pts` x `height_pts` points (72 points per inch).

    `crop_box` (width, height) adds a CropBox to every page, which pdfinfo reports in place of the MediaBox.
    `page_sizes` gives one (width, height) MediaBox per page and replaces `pages`, `width_pts` and `height_pts`.
    """
    if page_sizes is not None:
        pages = len(page_sizes)
    else:
        page_sizes = [(width_pts, height_pts)] * pages
    crop = f" /CropBox [0 0 {crop_box[0]} {crop_box[1]}]" if crop_box else ""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    for width, height in page_sizes:
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}]{crop} >>".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def png_header(width: int, height: int) -> bytes:
    """A PNG signature plus an IHDR chunk declaring `width` x `height`; there is no pixel data."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


def jpeg_header(width: int, height: int, progressive: bool = False, leading_segments: int = 1) -> bytes:
    """SOI, `leading_segments` APP1 segments (like EXIF), then a SOF0 (or SOF2) frame header declaring the size."""
    out = bytearray(b"\xff\xd8")
    for _ in range(leading_segments):
        payload = b"Exif\x00\x00" + b"\x00" * 20
        out += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    sof = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    out += (b"\xff\xc2" if progressive else b"\xff\xc0") + struct.pack(">H", len(sof) + 2) + sof
    return bytes(out) + b"\xff\xda\x00\x02"

# ABOUTME: Generates the synthetic EN/ES invoice corpus (born-digital PDFs, simulated scan/photo PNGs, ground-truth JSON).
# ABOUTME: Fully seeded and byte-reproducible; run with /usr/local/bin/python3 (needs Pillow, numpy, reportlab, pdftoppm).
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

SEED = 20261007
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
PDF_DIR = os.path.join(HERE, "pdf")
IMG_DIR = os.path.join(HERE, "images")
GT_DIR = os.path.join(HERE, "ground_truth")

N_EN = 18
N_ES = 12
LAYOUTS = ["classic", "letter", "twocol"]
CENT = Decimal("0.01")

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"],
}
MONTHS_SHORT_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

LABELS = {
    "en": {
        "invoice": "INVOICE", "number": "Invoice No.", "date": "Invoice Date", "due": "Due Date",
        "po": "PO Number", "bill_to": "Bill To", "from": "From", "desc": "Description",
        "qty": "Qty", "unit": "Unit Price", "amount": "Amount", "subtotal": "Subtotal",
        "tax": "Tax", "total": "Total", "terms": "Terms", "thanks": "Thank you for your business.",
        "intro": "Please find below the charges for the services and goods delivered.",
        "close": "Payment is due by the date shown above. Please quote the invoice number with your remittance.",
        "sign": "Accounts Receivable", "re": "Re: Invoice No.",
    },
    "es": {
        "invoice": "FACTURA", "number": "N.º de factura", "date": "Fecha de emisión", "due": "Vencimiento",
        "po": "Orden de compra", "bill_to": "Facturar a", "from": "Emisor", "desc": "Descripción",
        "qty": "Cant.", "unit": "Precio unit.", "amount": "Importe", "subtotal": "Subtotal",
        "tax": "IVA", "total": "Total", "terms": "Condiciones", "thanks": "Gracias por su preferencia.",
        "intro": "Adjuntamos el detalle de los cargos por los servicios y productos entregados.",
        "close": "El pago vence en la fecha indicada. Indique el número de factura en su transferencia.",
        "sign": "Cuentas por Cobrar", "re": "Ref.: Factura N.º",
    },
}

# Currency profiles: (code, symbol, number style, tax label rate options, page size)
CURRENCIES = {
    "USD": {"symbol": "$", "style": "us", "rates": [Decimal("0"), Decimal("0.0825"), Decimal("0.07")], "page": letter},
    "GBP": {"symbol": "£", "style": "us", "rates": [Decimal("0.2")], "page": A4},
    "MXN": {"symbol": "$", "style": "us", "rates": [Decimal("0.16")], "page": letter},
    "EUR": {"symbol": "€", "style": "eu", "rates": [Decimal("0.21"), Decimal("0.10")], "page": A4},
}

# Every vendor is fictional. The two entries with a vendor_id are the generic fictional
# vendors already present in data/vendor_master.json, reused so PO routing stays realistic.
# items: (description, unit price low, unit price high, qty low, qty high)
VENDORS = {
    "en": [
        {"name": "Northgate Ridge Supply Co.", "addr": ["4410 Fern Hollow Rd", "Brookfield, OH 44000"], "id": None,
         "cur": ["USD"], "num": "NR-{n:05d}", "color": (0.12, 0.30, 0.45),
         "items": [("Pallet wrap, 18in x 1500ft", 18, 34, 4, 40), ("Corrugated boxes, 24x18x12", 2, 5, 50, 400),
                   ("Packing tape, case of 36", 38, 52, 2, 10), ("Edge protectors, bundle", 14, 26, 5, 30)]},
        {"name": "Brightwater Janitorial LLC", "addr": ["88 Tamarack Lane", "Dunmore, PA 18500"], "id": None,
         "cur": ["USD"], "num": "BW{n:06d}", "color": (0.10, 0.45, 0.40),
         "items": [("Monthly cleaning service, Suite 200", 640, 1450, 1, 1), ("Floor stripping and waxing", 280, 520, 1, 2),
                   ("Restroom supply restock", 85, 160, 1, 3), ("Window cleaning, interior", 120, 240, 1, 2)]},
        {"name": "Copperline Logistics Inc.", "addr": ["1900 Wharf Street, Unit 7", "Savannah, GA 31401"], "id": None,
         "cur": ["USD"], "num": "CL-{n:04d}-26", "color": (0.55, 0.28, 0.12),
         "items": [("Freight, LTL Savannah to Columbus", 310, 780, 1, 3), ("Fuel surcharge", 42, 96, 1, 3),
                   ("Liftgate delivery fee", 65, 90, 1, 2), ("Storage, per pallet per week", 12, 18, 4, 20)]},
        {"name": "Harborview Print & Signage", "addr": ["27 Quay Road", "Portland, ME 04101"], "id": None,
         "cur": ["USD"], "num": "HP{n:05d}", "color": (0.35, 0.15, 0.40),
         "items": [("Vinyl banner 3x8ft", 85, 140, 1, 4), ("Business cards, box of 500", 32, 55, 2, 6),
                   ("Foam board posters 24x36", 22, 38, 5, 25), ("Installation labour, hourly", 70, 95, 2, 6)]},
        {"name": "Quillstone Software Services", "addr": ["500 Alder Court, Floor 3", "Austin, TX 78701"], "id": None,
         "cur": ["USD", "GBP"], "num": "QS-{n:06d}", "color": (0.20, 0.20, 0.55),
         "items": [("Platform subscription, monthly", 450, 1900, 1, 1), ("Additional seats", 24, 49, 5, 40),
                   ("Onboarding workshop, half day", 900, 1400, 1, 2), ("Priority support add-on", 150, 320, 1, 1)]},
        {"name": "Larkspur Catering Group", "addr": ["62 Orchard Way", "Madison, WI 53703"], "id": None,
         "cur": ["USD"], "num": "LC-{n:04d}", "color": (0.60, 0.20, 0.25),
         "items": [("Working lunch, per head", 14, 26, 10, 60), ("Coffee and pastry service", 9, 15, 10, 40),
                   ("Service staff, hourly", 28, 40, 4, 16), ("Equipment rental", 60, 140, 1, 2)]},
        {"name": "Meridian Fieldworks Ltd.", "addr": ["Unit 12, Cairn Business Park", "Leeds LS11 5BD"], "id": None,
         "cur": ["GBP"], "num": "MF/{n:05d}", "color": (0.25, 0.40, 0.20),
         "items": [("Site survey, per day", 380, 620, 1, 4), ("Soil sampling kit", 45, 90, 2, 10),
                   ("Reporting and drawings", 240, 480, 1, 2), ("Travel, per mile", 1, 1, 40, 220)]},
        {"name": "Tidewell Office Furniture", "addr": ["301 Kestrel Avenue", "Tacoma, WA 98402"], "id": None,
         "cur": ["USD"], "num": "TW-{n:05d}", "color": (0.30, 0.30, 0.30),
         "items": [("Height-adjustable desk", 340, 620, 1, 6), ("Task chair, mesh back", 160, 280, 1, 8),
                   ("Monitor arm, dual", 55, 95, 1, 8), ("Delivery and assembly", 120, 260, 1, 1)]},
        {"name": "Apex Consulting Group", "addr": ["1 Summit Plaza, Suite 900", "Chicago, IL 60601"], "id": "VEND-1003",
         "cur": ["USD"], "num": "AC-{n:05d}", "color": (0.15, 0.15, 0.15),
         "items": [("Advisory services, senior consultant (hrs)", 180, 260, 4, 30), ("Process review workshop", 1200, 2400, 1, 1),
                   ("Reporting and documentation (hrs)", 110, 150, 2, 12)]},
        {"name": "Acme Marketing Solutions", "addr": ["77 Mercer Street", "New York, NY 10012"], "id": "VEND-1005",
         "cur": ["USD"], "num": "AMS-{n:05d}", "color": (0.70, 0.35, 0.05),
         "items": [("Campaign management fee", 900, 2600, 1, 1), ("Creative design (hrs)", 85, 130, 5, 30),
                   ("Social media ad spend", 400, 1500, 1, 1), ("Landing page build", 750, 1800, 1, 2)]},
    ],
    "es": [
        {"name": "Ferretería Los Álamos S.A. de C.V.", "addr": ["Av. Revolución 1450, Col. Centro", "Guadalajara, Jal. 44100"], "id": None,
         "cur": ["MXN"], "num": "FA-{n:05d}", "color": (0.60, 0.10, 0.10),
         "items": [("Tornillería surtida, caja", 180, 420, 2, 12), ("Cable calibre 12, rollo 100 m", 950, 1380, 1, 4),
                   ("Pintura vinílica, cubeta 19 L", 1100, 1750, 1, 6), ("Herramienta menor, kit", 380, 760, 1, 5)]},
        {"name": "Distribuidora Costa Verde S.L.", "addr": ["Calle del Puerto 22, Local 3", "46001 Valencia"], "id": None,
         "cur": ["EUR"], "num": "DCV-2026/{n:04d}", "color": (0.10, 0.50, 0.30),
         "items": [("Aceite de oliva, caja 12 L", 62, 98, 2, 20), ("Conservas surtidas, caja", 28, 54, 4, 30),
                   ("Transporte refrigerado", 85, 190, 1, 2), ("Envases reutilizables", 6, 12, 10, 60)]},
        {"name": "Papelería El Roble S.A.", "addr": ["Calle 50 No. 12-34", "Bogotá D.C."], "id": None,
         "cur": ["USD"], "num": "PR-{n:06d}", "color": (0.40, 0.25, 0.10),
         "items": [("Resma de papel carta, caja 10", 38, 55, 2, 20), ("Carpetas archivadoras, paquete", 12, 24, 5, 40),
                   ("Tóner compatible, unidad", 45, 88, 1, 6), ("Marcadores surtidos, caja", 8, 16, 3, 20)]},
        {"name": "Servicios Tecnológicos Alborada S.A.", "addr": ["Paseo de la Reforma 505, Piso 12", "Ciudad de México 06500"], "id": None,
         "cur": ["MXN", "USD"], "num": "STA-{n:05d}", "color": (0.15, 0.25, 0.60),
         "items": [("Soporte técnico mensual", 4800, 12500, 1, 1), ("Mantenimiento de servidores (hrs)", 650, 980, 2, 14),
                   ("Licencia de antivirus, anual", 280, 540, 5, 40), ("Instalación de equipo", 900, 1900, 1, 4)]},
        {"name": "Imprenta Sol Naciente S.L.", "addr": ["Plaza Mayor 8, Bajo", "28012 Madrid"], "id": None,
         "cur": ["EUR"], "num": "ISN-{n:05d}", "color": (0.65, 0.45, 0.05),
         "items": [("Folletos A5 a color, millar", 78, 140, 1, 5), ("Diseño gráfico (hrs)", 38, 55, 2, 12),
                   ("Tarjetas de presentación, 500 u.", 24, 40, 1, 8), ("Lonas publicitarias 2x1 m", 32, 58, 1, 6)]},
        {"name": "Transportes Cumbre del Sur S.A.", "addr": ["Ruta 5 Sur Km 14", "Santiago, Región Metropolitana"], "id": None,
         "cur": ["USD"], "num": "TCS-{n:05d}", "color": (0.20, 0.30, 0.35),
         "items": [("Flete terrestre, por tonelada", 42, 78, 4, 30), ("Seguro de carga", 60, 150, 1, 2),
                   ("Maniobras de carga y descarga", 90, 210, 1, 3), ("Estadía por día", 120, 180, 1, 4)]},
        {"name": "Limpieza Integral Mirador S.L.", "addr": ["Calle Alcalá 140, 2.º B", "28009 Madrid"], "id": None,
         "cur": ["EUR"], "num": "LIM-{n:05d}", "color": (0.10, 0.55, 0.60),
         "items": [("Limpieza mensual de oficinas", 520, 1180, 1, 1), ("Limpieza de cristales", 95, 210, 1, 2),
                   ("Reposición de consumibles", 40, 110, 1, 3), ("Tratamiento de suelos", 180, 340, 1, 2)]},
        {"name": "Consultoría Estrella del Norte S.C.", "addr": ["Blvd. Díaz Ordaz 2300, Piso 4", "Monterrey, N.L. 64650"], "id": None,
         "cur": ["MXN"], "num": "CEN-{n:04d}", "color": (0.35, 0.15, 0.45),
         "items": [("Asesoría fiscal mensual", 6500, 14000, 1, 1), ("Auditoría de procesos (hrs)", 780, 1150, 4, 24),
                   ("Capacitación al personal, sesión", 3200, 5400, 1, 3), ("Informe ejecutivo", 2400, 4800, 1, 1)]},
    ],
}

BUYER = {
    "en": ("Cedar & Finch Operating Co.", ["200 Demo Boulevard", "Exampleton, ST 00000"]),
    "es": ("Cedar & Finch Operaciones S.A.", ["Calle Demostración 200", "Ejemplópolis 00000"]),
}


def fmt_money(value, currency):
    """Format a Decimal in the currency's display convention (us: $1,234.56, eu: 1.234,56 €)."""
    cur = CURRENCIES[currency]
    text = f"{value:,.2f}"
    if cur["style"] == "eu":
        text = text.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{text} {cur['symbol']}"
    return f"{cur['symbol']}{text}"


def fmt_date(iso, lang, style):
    y, m, d = (int(p) for p in iso.split("-"))
    if style == "iso":
        return iso
    if lang == "en":
        if style == "long":
            return f"{MONTHS['en'][m - 1]} {d}, {y}"
        return f"{d} {MONTHS_SHORT_EN[m - 1]} {y}"
    if style == "long":
        return f"{d} de {MONTHS['es'][m - 1]} de {y}"
    return f"{d:02d}/{m:02d}/{y}"


def q(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def build_invoice(doc_id, lang, vendor, layout, rng, po_db):
    currency = rng.choice(vendor["cur"])
    cur = CURRENCIES[currency]
    n_items = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
    n_items = min(n_items, len(vendor["items"]) + 1)
    picks = [rng.choice(vendor["items"]) for _ in range(n_items)]
    seen, line_items = set(), []
    for desc, lo, hi, qlo, qhi in picks:
        if desc in seen:
            continue
        seen.add(desc)
        qty = rng.randint(qlo, qhi)
        unit = q(Decimal(str(rng.uniform(lo, hi))))
        line_items.append({"description": desc, "quantity": qty, "unit_price": unit, "amount": q(unit * qty)})
    subtotal = sum((li["amount"] for li in line_items), Decimal("0"))
    rate = rng.choice(cur["rates"])
    tax = q(subtotal * rate)
    total = subtotal + tax

    month = rng.randint(1, 9)
    day = rng.randint(1, 28)
    invoice_date = f"2026-{month:02d}-{day:02d}"
    due_date = None
    terms = None
    if rng.random() < 0.7:
        days = rng.choice([15, 30, 45])
        terms = f"Net {days}" if lang == "en" else f"Neto {days} días"
        due_date = (date(2026, month, day) + timedelta(days=days)).isoformat()

    po_number = None
    if vendor["id"]:
        candidates = [k for k, v in po_db.items() if v["vendor_id"] == vendor["id"]]
        po_number = rng.choice(candidates)
    elif rng.random() < 0.5:
        po_number = f"PO-{rng.randint(7000, 7999)}" if lang == "en" else f"OC-2026-{rng.randint(1000, 4999)}"

    return {
        "doc_id": doc_id,
        "vendor": vendor,
        "layout": layout,
        "language": lang,
        "currency": currency,
        "invoice_number": vendor["num"].format(n=rng.randint(1000, 9999) if "{n:04d}" in vendor["num"] else rng.randint(10000, 99999)),
        "invoice_date": invoice_date,
        "due_date": due_date,
        "terms": terms,
        "date_style": rng.choice(["long", "short", "iso"]),
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": rate,
        "tax": tax,
        "total": total,
        "po_number": po_number,
        "page": cur["page"],
    }


# ----------------------------------------------------------------------------- drawing

def bold_of(font):
    return "Times-Bold" if font == "Times-Roman" else font + "-Bold"


def fits(text, font, size, max_width):
    width = stringWidth(text, font, size)
    if width > max_width:
        raise ValueError(f"text overflows ({width:.0f} > {max_width:.0f}pt): {text!r}")
    return text


def tax_label(inv):
    rate = (inv["tax_rate"] * 100).normalize()
    label = LABELS[inv["language"]]["tax"]
    return f"{label} ({rate:f}%)"


def draw_items(c, inv, x0, x1, y, grid, header_fill=None, font="Helvetica", size=9.5, row_h=20):
    """Draw header + rows; returns the y below the table. Columns: desc | qty | unit | amount."""
    lab = LABELS[inv["language"]]
    cur = inv["currency"]
    amount_r, unit_r, qty_r = x1 - 6, x1 - 96, x1 - 176
    desc_x = x0 + 6
    desc_w = (qty_r - 40) - desc_x
    bold = bold_of(font)
    if header_fill is not None:
        c.setFillColor(header_fill)
        c.rect(x0, y - row_h + 6, x1 - x0, row_h, stroke=0, fill=1)
        c.setFillColor(colors.white)
    else:
        c.setFillColor(colors.black)
    c.setFont(bold, size)
    c.drawString(desc_x, y - 8, lab["desc"])
    c.drawRightString(qty_r, y - 8, lab["qty"])
    c.drawRightString(unit_r, y - 8, lab["unit"])
    c.drawRightString(amount_r, y - 8, lab["amount"])
    c.setFillColor(colors.black)
    top = y + 6
    y -= row_h
    if not grid:
        c.setLineWidth(0.8)
        c.line(x0, y + 6, x1, y + 6)
    c.setFont(font, size)
    for li in inv["line_items"]:
        fits(li["description"], font, size, desc_w)
        c.drawString(desc_x, y - 8, li["description"])
        c.drawRightString(qty_r, y - 8, str(li["quantity"]))
        c.drawRightString(unit_r, y - 8, fmt_money(li["unit_price"], cur))
        c.drawRightString(amount_r, y - 8, fmt_money(li["amount"], cur))
        y -= row_h
        if grid:
            c.setStrokeColor(colors.Color(0.75, 0.75, 0.75))
            c.setLineWidth(0.5)
            c.line(x0, y + 6, x1, y + 6)
            c.setStrokeColor(colors.black)
    if grid:
        c.setLineWidth(0.8)
        c.rect(x0, y + 6, x1 - x0, top - (y + 6), stroke=1, fill=0)
    return y


def draw_totals(c, inv, x_label, x_right, y, font="Helvetica", size=10.5):
    lab = LABELS[inv["language"]]
    cur = inv["currency"]
    rows = [(lab["subtotal"], fmt_money(inv["subtotal"], cur)), (tax_label(inv), fmt_money(inv["tax"], cur))]
    c.setFillColor(colors.black)
    for label, value in rows:
        c.setFont(font, size)
        c.drawString(x_label, y, label)
        c.drawRightString(x_right, y, value)
        y -= 16
    c.setFont(bold_of(font), size + 1.5)
    c.drawString(x_label, y - 2, f"{lab['total']} ({cur})")
    c.drawRightString(x_right, y - 2, fmt_money(inv["total"], cur))
    return y - 22


def meta_pairs(inv):
    lab = LABELS[inv["language"]]
    lang = inv["language"]
    pairs = [(lab["number"], inv["invoice_number"]), (lab["date"], fmt_date(inv["invoice_date"], lang, inv["date_style"]))]
    if inv["due_date"]:
        pairs.append((lab["due"], fmt_date(inv["due_date"], lang, inv["date_style"])))
    if inv["po_number"]:
        pairs.append((lab["po"], inv["po_number"]))
    return pairs


def draw_classic(c, inv, W, H):
    v, lab = inv["vendor"], LABELS[inv["language"]]
    r, g, b = v["color"]
    c.setFillColor(colors.Color(r, g, b))
    c.rect(0, H - 96, W, 96, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(48, H - 46, fits(v["name"], "Helvetica-Bold", 20, W - 250))
    c.setFont("Helvetica", 9)
    c.drawString(48, H - 62, v["addr"][0])
    c.drawString(48, H - 74, v["addr"][1])
    c.setFont("Helvetica-Bold", 24)
    c.drawRightString(W - 48, H - 50, lab["invoice"])
    y = H - 130
    c.setFillColor(colors.black)
    for label, value in meta_pairs(inv):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(48, y, label + ":")
        c.setFont("Helvetica", 10)
        c.drawString(150, y, value)
        y -= 16
    buyer, baddr = BUYER[inv["language"]]
    bx = W - 48 - 190
    c.setLineWidth(0.8)
    c.rect(bx, H - 190, 190, 72, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(bx + 8, H - 130, lab["bill_to"].upper())
    c.setFont("Helvetica", 10)
    c.drawString(bx + 8, H - 146, fits(buyer, "Helvetica", 10, 174))
    c.drawString(bx + 8, H - 160, baddr[0])
    c.drawString(bx + 8, H - 174, baddr[1])
    y = draw_items(c, inv, 48, W - 48, min(y, H - 190) - 28, grid=True, header_fill=colors.Color(r, g, b))
    y = draw_totals(c, inv, W - 48 - 230, W - 54, y - 22)
    c.setFont("Helvetica-Oblique", 9)
    if inv["terms"]:
        c.drawString(48, 90, f"{lab['terms']}: {inv['terms']}")
    c.drawString(48, 72, lab["thanks"])


def draw_letter(c, inv, W, H):
    v, lab, lang = inv["vendor"], LABELS[inv["language"]], inv["language"]
    c.setFont("Times-Bold", 17)
    c.drawString(60, H - 70, fits(v["name"], "Times-Bold", 17, W - 120))
    c.setFont("Times-Roman", 10.5)
    c.drawString(60, H - 86, v["addr"][0])
    c.drawString(60, H - 99, v["addr"][1])
    c.drawRightString(W - 60, H - 70, fmt_date(inv["invoice_date"], lang, inv["date_style"]))
    buyer, baddr = BUYER[lang]
    y = H - 150
    c.setFont("Times-Roman", 11)
    for line in [buyer, baddr[0], baddr[1]]:
        c.drawString(60, y, line)
        y -= 14
    y -= 14
    c.setFont("Times-Bold", 12)
    c.drawString(60, y, f"{lab['re']} {inv['invoice_number']}")
    y -= 16
    c.setFont("Times-Roman", 11)
    if inv["po_number"]:
        c.drawString(60, y, f"{lab['po']}: {inv['po_number']}")
        y -= 14
    if inv["due_date"]:
        c.drawString(60, y, f"{lab['due']}: {fmt_date(inv['due_date'], lang, inv['date_style'])}")
        y -= 14
    y -= 10
    c.drawString(60, y, lab["intro"])
    y -= 30
    y = draw_items(c, inv, 60, W - 60, y, grid=False, font="Times-Roman", size=10.5, row_h=18)
    y = draw_totals(c, inv, W - 60 - 220, W - 60, y - 20, font="Times-Roman", size=11)
    c.setFont("Times-Roman", 10.5)
    c.drawString(60, y - 18, lab["close"])
    c.drawString(60, y - 48, lab["sign"])
    c.drawString(60, y - 62, v["name"])


def draw_twocol(c, inv, W, H):
    v, lab, lang = inv["vendor"], LABELS[inv["language"]], inv["language"]
    r, g, b = v["color"]
    side_w = 178
    c.setFillColor(colors.Color(0.5 + r / 2, 0.5 + g / 2, 0.5 + b / 2, alpha=1))
    c.rect(0, 0, side_w, H - 110, stroke=0, fill=1)
    c.setFillColor(colors.Color(r, g, b))
    c.rect(0, H - 110, W, 110, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(28, H - 64, lab["invoice"])
    c.setFont("Helvetica-Bold", 13)
    c.drawRightString(W - 36, H - 56, fits(v["name"], "Helvetica-Bold", 13, W - side_w - 60))
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 36, H - 71, v["addr"][0])
    c.drawRightString(W - 36, H - 83, v["addr"][1])
    c.setFillColor(colors.black)
    y = H - 140
    for label, value in meta_pairs(inv):
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(20, y, label.upper())
        c.setFont("Helvetica", 10)
        c.drawString(20, y - 13, fits(value, "Helvetica", 10, side_w - 30))
        y -= 38
    buyer, baddr = BUYER[lang]
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(20, y, lab["bill_to"].upper())
    c.setFont("Helvetica", 9)
    c.drawString(20, y - 13, fits(buyer, "Helvetica", 9, side_w - 26))
    c.drawString(20, y - 25, baddr[0])
    c.drawString(20, y - 37, baddr[1])
    if inv["terms"]:
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(20, y - 62, lab["terms"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(20, y - 75, inv["terms"])
    x0 = side_w + 24
    y = draw_items(c, inv, x0, W - 36, H - 150, grid=False, size=9, row_h=20)
    y = draw_totals(c, inv, W - 36 - 210, W - 42, y - 20)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(x0, 60, lab["thanks"])


DRAWERS = {"classic": draw_classic, "letter": draw_letter, "twocol": draw_twocol}


def render_pdf(inv, path):
    W, H = inv["page"]
    # invariant=1 pins creation date and document ID so identical input gives identical bytes.
    c = canvas.Canvas(path, pagesize=inv["page"], invariant=1, pageCompression=1)
    c.setTitle(f"{inv['doc_id']}")
    c.setAuthor("Synthetic corpus generator")
    DRAWERS[inv["layout"]](c, inv, W, H)
    c.showPage()
    c.save()


# ----------------------------------------------------------------------------- image degradation

def rasterise(pdf_path, out_png):
    """Render page 1 at 150 dpi via pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "page")
        result = subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", pdf_path, prefix],
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed ({result.returncode}): {result.stderr}")
        shutil.copyfile(prefix + ".png", out_png)


def degrade(png_path, variant, np_rng):
    """Simulate a flatbed scan or a handheld photo using local image ops only."""
    img = Image.open(png_path).convert("L")
    w, h = img.size
    if variant == "scan":
        angle = float(np_rng.uniform(-1.2, 1.2))
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=242)
        blur, sigma, contrast, brightness = 0.6, 3.5, 0.92, -6
    else:
        coeffs = (1.0, float(np_rng.uniform(-0.02, 0.02)), 0.0,
                  float(np_rng.uniform(-0.015, 0.015)), 1.0, 0.0,
                  float(np_rng.uniform(-2.5e-5, 2.5e-5)), float(np_rng.uniform(-2.0e-5, 2.0e-5)))
        img = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=96)
        angle = float(np_rng.uniform(-2.2, 2.2))
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=96)
        blur, sigma, contrast, brightness = 0.9, 5.0, 0.82, -14
    img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32)
    if variant == "photo":
        # Uneven lighting: a diagonal gradient plus a soft vignette.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gradient = 1.0 - 0.14 * (xx / w) - 0.08 * (yy / h)
        vignette = 1.0 - 0.18 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
        arr = arr * gradient * vignette
    arr = (arr - 128.0) * contrast + 128.0 + brightness
    arr = arr + np_rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    out = out.filter(ImageFilter.GaussianBlur(0.35))
    out.save(png_path, format="PNG", optimize=True, compress_level=9)


# ----------------------------------------------------------------------------- ground truth

def ground_truth(inv, doc_id, variant, base_doc_id, source_file):
    li = [{"description": x["description"], "quantity": x["quantity"],
           "unit_price": float(x["unit_price"]), "amount": float(x["amount"])} for x in inv["line_items"]]
    rec = {
        "doc_id": doc_id,
        "base_doc_id": base_doc_id,
        "source_file": source_file,
        "language": inv["language"],
        "variant": variant,
        "layout": inv["layout"],
        "vendor_name": inv["vendor"]["name"],
        "vendor_id": inv["vendor"]["id"],
        "invoice_number": inv["invoice_number"],
        "invoice_date": inv["invoice_date"],
        "due_date": inv["due_date"],
        "currency": inv["currency"],
        "subtotal": float(inv["subtotal"]),
        "tax_rate": float(inv["tax_rate"]),
        "tax": float(inv["tax"]),
        "total": float(inv["total"]),
        "total_display": fmt_money(inv["total"], inv["currency"]),
        "po_number": inv["po_number"],
        "line_items": li,
    }
    if variant in ("scan", "photo"):
        rec["ocr_lang"] = "eng"
    return rec


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    rng = random.Random(SEED)
    with open(os.path.join(DATA_DIR, "po_database.json"), encoding="utf-8") as f:
        po_db = json.load(f)

    for d in (PDF_DIR, IMG_DIR, GT_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    plan = []  # (doc_id, lang, vendor, layout)
    for lang, count in (("en", N_EN), ("es", N_ES)):
        layouts = [LAYOUTS[i % len(LAYOUTS)] for i in range(count)]
        rng.shuffle(layouts)
        vendors = [VENDORS[lang][i % len(VENDORS[lang])] for i in range(count)]
        rng.shuffle(vendors)
        for i in range(count):
            plan.append((f"{lang}-{i + 1:03d}", lang, vendors[i], layouts[i]))

    # Image variants: ~a third of the corpus. 4 scan + 2 photo EN, 2 scan + 2 photo ES.
    en_ids = [p[0] for p in plan if p[1] == "en"]
    es_ids = [p[0] for p in plan if p[1] == "es"]
    en_pick = rng.sample(en_ids, 6)
    es_pick = rng.sample(es_ids, 4)
    variant_of = {}
    for ids, n_scan in ((en_pick, 4), (es_pick, 2)):
        for i, doc_id in enumerate(sorted(ids)):
            variant_of[doc_id] = "scan" if i < n_scan else "photo"
    # Interleave so scans and photos are not just "first N sorted ids".
    for ids in (en_pick, es_pick):
        kinds = [variant_of[i] for i in sorted(ids)]
        rng.shuffle(kinds)
        for doc_id, kind in zip(sorted(ids), kinds):
            variant_of[doc_id] = kind

    for index, (doc_id, lang, vendor, layout) in enumerate(plan):
        inv = build_invoice(doc_id, lang, vendor, layout, rng, po_db)
        pdf_path = os.path.join(PDF_DIR, f"{doc_id}.pdf")
        render_pdf(inv, pdf_path)
        write_json(os.path.join(GT_DIR, f"{doc_id}.json"),
                   ground_truth(inv, doc_id, "pdf", doc_id, f"pdf/{doc_id}.pdf"))
        variant = variant_of.get(doc_id)
        if variant:
            img_id = f"{doc_id}-{variant}"
            img_path = os.path.join(IMG_DIR, f"{img_id}.png")
            rasterise(pdf_path, img_path)
            degrade(img_path, variant, np.random.default_rng(SEED + index))
            write_json(os.path.join(GT_DIR, f"{img_id}.json"),
                       ground_truth(inv, img_id, variant, doc_id, f"images/{img_id}.png"))
        print(f"generated {doc_id} ({lang}, {layout}, {variant or 'pdf only'})")

    digest = hashlib.sha256()
    for sub in ("pdf", "images", "ground_truth"):
        for name in sorted(os.listdir(os.path.join(HERE, sub))):
            with open(os.path.join(HERE, sub, name), "rb") as f:
                digest.update(name.encode() + f.read())
    print(f"corpus sha256: {digest.hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())

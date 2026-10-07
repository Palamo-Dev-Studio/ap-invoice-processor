# Synthetic invoice corpus

All documents here are **synthetic**. Every vendor, address, buyer and amount is invented (two vendors,
Apex Consulting Group and Acme Marketing Solutions, are the generic fictional entries already in
`data/vendor_master.json`, reused so their POs match `data/po_database.json`). No real companies or
client data are involved.

## Contents

- `pdf/<doc_id>.pdf` — 30 born-digital invoices (18 EN `en-001..018`, 12 ES `es-001..012`) in three layouts:
  `classic` (banded header, gridded table), `letter` (letter-style, no grid), `twocol` (header block + tinted sidebar).
  Some have tax, some a PO number, some a due date; currencies are USD, GBP, EUR and MXN.
- `images/<doc_id>-<scan|photo>.png` — 10 simulated scan/photo variants of some PDFs (rendered with
  `pdftoppm -r 150`, then rotation/perspective skew, blur, noise and contrast/lighting shifts via Pillow and numpy).
- `ground_truth/<doc_id>.json` — one record per document (PDF or image): vendor, invoice number, ISO dates,
  currency, subtotal, tax, total, `total_display` (the total as printed), PO number or null, line items,
  language, variant, layout. Image records also carry `"ocr_lang": "eng"`.

## OCR language caveat

The installed Tesseract data holds only `eng` and `osd`. Spanish scan/photo variants are therefore OCR'd
with `-l eng` and will show accent damage; do not quote a Spanish OCR accuracy figure from this corpus.

## Ground-truth fields vs `ap_invoice_processor/models.py`

| Ground truth | Model field |
|---|---|
| `vendor_name` | `ExtractedInvoiceFields.vendor_name` |
| `vendor_id` | `ExtractedInvoiceFields.vendor_id` |
| `invoice_number` | `ExtractedInvoiceFields.invoice_number` |
| `invoice_date` (ISO) | `ExtractedInvoiceFields.date` |
| `po_number` | `ExtractedInvoiceFields.po_number` |
| `total` | `ExtractedInvoiceFields.total_amount` |
| `line_items[].description` | `LineItem.description` |
| `line_items[].quantity` | `LineItem.qty` |
| `line_items[].unit_price` | `LineItem.unit_price` |
| `line_items[].amount` | `LineItem.amount` |

Ground-truth fields with no model counterpart: `due_date`, `currency`, `subtotal`, `tax`, `tax_rate`,
`total_display` (the printed total, used by the reader tests), `language`, `variant`, `layout`, `ocr_lang`,
`base_doc_id`, `source_file`, `doc_id`.

## OCR coverage on the image variants

The reader OCRs each image twice (Tesseract default plus Sauvola thresholding, `-c thresholding_method=2`)
and joins the outputs, because the default pass alone drops the tinted sidebar of the `twocol` layout.
`tests/test_reader.py` asserts that all 10 image variants yield the ground-truth invoice number and
`total_display`; all 10 pass. Fields that are **not** reliably recovered by OCR:

- Vendor name: missing from the OCR text for `en-017-photo`, `es-002-scan` and `es-011-photo` (image
  degradation, not only accents), and for `es-003-photo` and `es-007-scan` only because accents are damaged
  (they match once accents are folded).
- Spanish accents in general: OCR runs with `-l eng` (only `eng` and `osd` are installed).
- Invoice date in the printed form is not asserted for images; the ground truth stores it as ISO.

## GL labels

`gl_labels.json` maps each `doc_id` (all 40 documents) to its line items, each with the expected GL account:
`{"line": <index>, "description": ..., "account": "<chart account number>" | null, "basis": "<short reason>"}`.

Labelling rule: each line description was read against the account names in `data/gl_chart_of_accounts.json`
and given the one account whose name covers the expense the line describes (marketing and advertising
materials, software licences and subscriptions, office furniture and stationery, consulting, reporting and IT
technical services). Ancillary charges (delivery, assembly) follow the goods they relate to. `account` is
`null` when no account in the five-account chart fits (freight, warehousing, catering, janitorial, food goods,
hardware-store goods, field kits and travel); 42 of the 98 lines are `null`. The labels were written by hand
from the line descriptions before any coder or fixture for them existed; no coder was run to produce them, and
they are one reader's reading of a small chart, not an audited standard. An image variant carries the same
lines as its base document, so its labels repeat.

## Regenerate

```
/usr/local/bin/python3 data/corpus/generate_corpus.py
```

Needs Pillow, numpy and reportlab (the project venv does not have them) and `pdftoppm` on PATH. Every random
operation is seeded (`SEED` in the script), so reruns reproduce the committed files byte for byte; the script
prints a corpus-wide sha256 at the end. The script deletes and rewrites `pdf/`, `images/` and `ground_truth/`.

Byte-for-byte regeneration also depends on the installed library versions. The committed files were produced with
reportlab 4.5.1, Pillow 12.1.1, numpy 2.4.3 and poppler `pdftoppm` 26.09.0 (Python 3.13.7). Other versions
may change fonts, PDF object layout, rasterisation or the random-number-driven pixel noise, so the corpus-wide sha256
can differ even though every operation is seeded.

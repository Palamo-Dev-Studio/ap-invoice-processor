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

## Regenerate

```
/usr/local/bin/python3 data/corpus/generate_corpus.py
```

Needs Pillow, numpy and reportlab (the project venv does not have them) and `pdftoppm` on PATH. Every random
operation is seeded (`SEED` in the script), so reruns reproduce the committed files byte for byte; the script
prints a corpus-wide sha256 at the end. The script deletes and rewrites `pdf/`, `images/` and `ground_truth/`.

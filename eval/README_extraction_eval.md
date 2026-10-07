# Offline extraction eval

`eval/extraction_eval.py` runs the document pipeline end to end over the 40-document synthetic corpus and
scores it. It is a **plumbing check**, not a model evaluation. It is separate from `eval/eval_harness.py`, which
it does not touch.

## Run

```
PYTHONPATH=. .venv/bin/python eval/extraction_eval.py [--provider fixture] [--fixtures-dir DIR] [--out DIR]
```

- `--provider` defaults to `AP_LLM_PROVIDER`, then `fixture`. Only `fixture` runs; any other provider is refused
  with exit code 2 (see "Live provider" below).
- `--fixtures-dir` is a directory holding `extract/<doc_id>.json` and `gl/<doc_id>.json` (default
  `tests/fixtures/llm`).
- `--out` is where `extraction_eval.txt` and `extraction_eval.json` are written (default `eval/out/`, which is
  git-ignored: result numbers are never committed).

Needs `pdftotext` and `tesseract` on PATH (the reader shells out to them). No network, no new dependencies.

## What it does

1. `read_document` reads each PDF or image to text (`pdftotext`, or Tesseract with `-l eng`).
2. `extract_invoice` gets **only the `ReaderOutput`** and returns the extracted fields.
3. `code_lines` gets only the extracted lines, the chart of accounts and the extracted vendor name.
4. `eval/extraction_scoring.py` (the only module that opens ground truth and `gl_labels.json`) scores the result.

Ground truth, `simulated_extraction` and GL labels never reach the provider. `tests/test_extraction_eval.py`
records every prompt the provider receives and fails if a ground-truth-only value appears in one, and fails if the
pipeline module names or opens ground truth or labels.

## What it measures

- **Header fields** (vendor, invoice number, invoice and due date, currency, PO, subtotal, tax, total): scored on
  cells where ground truth holds a value. A value returned where ground truth is null is counted in its own column.
- **Line items**: description, quantity, unit price, amount, matched to ground-truth lines by description, then by
  position.
- Line-item cells whose ground-truth value is null still count in `n` and never match, unlike header cells, which
  are not scorable when ground truth is null. The corpus has no null line-item cells today.
- **Strict column**: amounts compared as decimals to the cent, dates as ISO strings, vendor/invoice/PO and
  descriptions case- and whitespace-insensitive with accents kept. **Lenient column**: the same with accents folded.
  The two are reported side by side and never merged.
- **Breakdowns** per language (en/es) and per variant (pdf/scan/photo).
- **GL coding** of the extracted lines, scored only on lines whose label is non-null. Null-label lines are counted
  in their own row ("unscorable (no fitting account)"). A line that fell back to the keyword coder is counted and
  listed, not scored; the keyword coder's own result against the labels is deliberately not computed, and the
  fallback record in the JSON carries no `got` (only `expected` and `fallback: true`). GL results are printed as
  counts only (for example "N/M scorable lines matched"), never as percentages, in stdout, the `.txt` file and the
  `.json` file; extraction keeps its rates.

## Disclosure

Every report (stdout, the `.txt` file, and the first key of the `.json` file) starts with a banner: the run is
offline and fixture-backed, the fixtures are hand-authored from reader text, no model ran, the numbers are not model
accuracy, GL fixture and label agreement is not independent (same author), and Spanish scans and photos are OCR'd
with `-l eng`. Do not quote numbers from a fixture run as AI accuracy anywhere.

## Negative controls

The tests generate, in a temporary directory, a blank fixture set (all-null and all-empty), a perturbed set (every
value changed; GL always a wrong account), a shifted set (each document gets the fixtures of another invoice, never
a pdf/scan/photo variant of its own) and an oracle set (copied from ground truth). Blank and perturbed must score at
or below 0.05 on extraction and GL; the real fixtures must score strictly higher than every control, and the oracle
must reach 1.0. This shows the scorer tells good from bad, which is all a fixture run can show. Shifted fixtures
must score below a quarter of the real extraction rate and below half of the real GL rate: with a donor from a
different invoice, what remains is coincidence (shared currencies, quantities, chart accounts). A donor that is a
variant of the same invoice would score high for a different reason, so the donor is chosen to avoid that.

## Live provider (later)

Once a provider is chosen: implement its `complete(task, prompt, doc_id)` in `ap_invoice_processor/llm/provider.py`
(the `AnthropicProvider` stub raises today) and select it with `--provider` or `AP_LLM_PROVIDER`. A live run needs
a disclosure banner written for a live run, set in `banner_for`, which refuses any provider other than the fixture
one so a live result can never carry the plumbing-check wording. A live score would still be limited by the corpus:
it is synthetic, the labels are one reader's reading of a five-account chart, and Spanish OCR needs `spa` language
data to be meaningful.

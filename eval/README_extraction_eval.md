# Offline extraction eval

`eval/extraction_eval.py` runs the document pipeline end to end over the 40-document synthetic corpus and
scores it. It is a **plumbing check**, not a model evaluation. It is separate from `eval/eval_harness.py`, which
it does not touch.

## Run

```
PYTHONPATH=. .venv/bin/python eval/extraction_eval.py [--provider fixture] [--fixtures-dir DIR] [--out DIR]
```

- `--provider` defaults to `AP_LLM_PROVIDER`, then `fixture`. `fixture` is the offline plumbing check described
  below; `anthropic` is the separate live-model run (see "Live-model run" below). Any other name, a missing key
  or cap, or a missing `--max-usd` with a live provider is refused with exit code 2.
- `--max-usd N` caps the spend of one live run in USD. It is required with a live provider and refused with the
  fixture provider.
- `--fixtures-dir` is a directory holding `extract/<doc_id>.json` and `gl/<doc_id>.json` (default
  `tests/fixtures/llm`).
- `--out` is where `extraction_eval.txt` and `extraction_eval.json` are written (default `eval/out/`, which is
  git-ignored: result numbers are never committed).

Needs `pdftotext` and `tesseract` on PATH (the reader shells out to them). A fixture run uses no network.

## What it does

1. `read_document` reads each PDF or image to text (`pdftotext`, or Tesseract with `eng+spa`; see `AP_OCR_LANGS`).
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
accuracy, GL fixture and label agreement is not independent (same author), and the fixtures were authored from reader
text of Spanish scans and photos OCR'd with `-l eng` (the reader now defaults to `eng+spa`). Do not quote numbers from a
fixture run as AI accuracy anywhere.

## Negative controls

The tests generate, in a temporary directory, a blank fixture set (all-null and all-empty), a perturbed set (every
value changed; GL always a wrong account), a shifted set (each document gets the fixtures of another invoice, never
a pdf/scan/photo variant of its own) and an oracle set (copied from ground truth). Blank and perturbed must score at
or below 0.05 on extraction and GL; the real fixtures must score strictly higher than every control, and the oracle
must reach 1.0. This shows the scorer tells good from bad, which is all a fixture run can show. Shifted fixtures
must score below a quarter of the real extraction rate and below half of the real GL rate: with a donor from a
different invoice, what remains is coincidence (shared currencies, quantities, chart accounts). A donor that is a
variant of the same invoice would score high for a different reason, so the donor is chosen to avoid that.

## Live-model run

```
PYTHONPATH=. .venv/bin/python eval/extraction_eval.py --provider anthropic --max-usd 1
```

This sends the corpus text to Claude Haiku 5.5 (needs `ANTHROPIC_API_KEY` and `AP_LLM_SPEND_CAP_USD`; see the
README's "Provider" paragraph) and scores what comes back with the same scorer. It is a different report from the
plumbing check:

- Its banner, its first section heading ("LIVE MODEL ACCURACY") and its files differ. It writes
  `extraction_eval_live.txt` and `extraction_eval_live.json`, never `extraction_eval.txt` or `.json`.
- The integrity rules are unchanged: the model receives only reader text (the same prompts, isolation tests
  included); ground truth and GL labels are opened only by the scorer after the run; GL results are counts, never
  percentages; lines with no fitting account are reported as unscorable; a GL line the keyword coder handled is
  counted and listed, not scored; and no LLM-versus-keyword figure is produced.
- A section 5 reports the model calls made, the spend of this run, the ledger total and the failures by stage
  (reader, provider, validation). A failed document stays in every denominator as a miss.
- `--max-usd` caps this run; `AP_LLM_SPEND_CAP_USD` caps the running total across runs
  (`eval/out/spend.json`, gitignored). If either would be passed, the run stops before sending the call that would
  pass it, the document in progress is dropped whole, the report opens with "INCOMPLETE RUN" and lists the
  documents not run, and the exit code is 3.
- A live score is limited by the corpus: it is synthetic, the GL labels are one reader's reading of a five-account
  chart (and a number of lines have no fitting account), and one run of one model is not a general accuracy claim.

`tests/test_extraction_eval_live.py` runs this path against the real provider with the HTTP layer mocked; the one
opt-in test that calls the API (`tests/test_llm_live.py`) runs only with `AP_LIVE=1`.

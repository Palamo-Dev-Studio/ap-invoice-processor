# ABOUTME: Opt-in live smoke test: one real extraction call to the Anthropic API, skipped unless AP_LIVE=1.
# ABOUTME: Spends a few cents under a per-run cap and writes to the gitignored spend ledger; never runs in CI.
import os

import pytest

from ap_invoice_processor.llm.extraction import extract_invoice
from ap_invoice_processor.llm.provider import get_provider
from ap_invoice_processor.reader import ReaderOutput

READER_TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "reader_text", "en-001.txt")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("AP_LIVE") != "1", reason="live API test: set AP_LIVE=1 to run it (spends real money, capped)"),
]


def test_one_real_extraction_round_trip_is_validated_and_priced():
    with open(READER_TEXT, encoding="utf-8") as f:
        text = f.read()
    provider = get_provider("anthropic", max_usd=0.05)
    invoice = extract_invoice(ReaderOutput(doc_id="live-smoke", text=text, method="pdftotext"), provider)
    assert invoice.invoice_number and invoice.invoice_number in text
    assert invoice.total > 0 and invoice.line_items
    assert provider.tracker.run_spent_usd > 0
    assert provider.tracker.run_spent_usd <= provider.tracker.run_cap_usd

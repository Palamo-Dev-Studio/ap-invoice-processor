# ABOUTME: Optional document intake: reads a PDF/image, extracts fields through the configured LLM provider, fills InvoiceState.
# ABOUTME: A failed read or extraction leaves empty fields at zero confidence so the policy validator routes to human review.
import os
from typing import Any, Dict, Optional

from ap_invoice_processor.llm.extraction import ExtractionError, extract_invoice
from ap_invoice_processor.llm.provider import LLMProvider, get_provider
from ap_invoice_processor.models import ExtractedInvoiceFields, FieldConfidence, InvoiceState, LineItem
from ap_invoice_processor.reader import ReaderError, read_document


def fill_state_from_document(
    invoice_state: InvoiceState, document_path: str, provider: Optional[LLMProvider] = None
) -> Dict[str, Any]:
    """Read `document_path`, extract fields, and write them into `invoice_state`.

    Returns a summary dict for the decision trail (its `doc_id` is the document's file stem). Field confidence is 1.0 for a field the extraction returned and
    0.0 for one it left null (or for everything when reading or extraction failed); no model confidence is available.
    """
    provider = provider or get_provider()
    summary: Dict[str, Any] = {
        "document_path": document_path,
        "doc_id": os.path.splitext(os.path.basename(document_path))[0],
        "provider": type(provider).__name__,
    }
    try:
        reader_output = read_document(document_path)
    except ReaderError as exc:
        _mark_failed(invoice_state)
        summary.update(extraction="error", error=f"reader: {exc}")
        return summary

    summary["reader_method"] = reader_output.method
    invoice_state.raw_text = reader_output.text
    try:
        extracted = extract_invoice(reader_output, provider)
    except ExtractionError as exc:
        _mark_failed(invoice_state)
        summary.update(extraction="error", error=exc.to_dict())
        return summary

    line_items = []
    for item in extracted.line_items:
        if float(item.quantity).is_integer():
            qty, unit_price = int(item.quantity), item.unit_price
        else:  # LineItem.qty is an int; carry a fractional quantity as one unit at the line amount.
            qty, unit_price = 1, item.amount
        line_items.append(LineItem(description=item.description, qty=qty, unit_price=unit_price, amount=item.amount))
    invoice_state.extracted_fields = ExtractedInvoiceFields(
        vendor_name=extracted.vendor_name,
        invoice_number=extracted.invoice_number,
        date=extracted.invoice_date,
        po_number=extracted.po_number,
        total_amount=extracted.total,
        line_items=line_items,
    )
    invoice_state.field_confidence = FieldConfidence(
        vendor_name=1.0 if extracted.vendor_name else 0.0,
        invoice_number=1.0 if extracted.invoice_number else 0.0,
        date=1.0 if extracted.invoice_date else 0.0,
        total_amount=1.0,
        line_items=1.0 if line_items else 0.0,
    )
    summary.update(extraction="ok", line_items=len(line_items))
    return summary


def _mark_failed(invoice_state: InvoiceState) -> None:
    invoice_state.extracted_fields = ExtractedInvoiceFields()
    invoice_state.field_confidence = FieldConfidence(
        vendor_name=0.0, invoice_number=0.0, date=0.0, total_amount=0.0, line_items=0.0
    )

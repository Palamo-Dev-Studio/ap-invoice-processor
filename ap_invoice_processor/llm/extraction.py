# ABOUTME: Invoice field extraction through an LLMProvider: builds the prompt from reader text only and validates the reply.
# ABOUTME: Invalid or missing responses become a structured ExtractionError; ground-truth inputs are rejected outright.
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ap_invoice_processor.llm.coercion import (
    optional_text,
    parse_amount,
    parse_currency,
    parse_date,
    parse_quantity,
)
from ap_invoice_processor.llm.provider import LLMProvider, LLMProviderError
from ap_invoice_processor.reader import ReaderOutput

TASK = "extract"
# The only fields extraction may see. Anything else (notably ground_truth or simulated_extraction) is rejected.
ALLOWED_INPUT_KEYS = frozenset(ReaderOutput.model_fields)

INSTRUCTIONS = """You extract structured fields from the text of one invoice.
Use only the document text between the markers. Do not guess: if a field is not printed or not legible, use null.
Return a single JSON object with exactly these keys:
  vendor_name (string or null), invoice_number (string or null),
  invoice_date (YYYY-MM-DD or null), due_date (YYYY-MM-DD or null),
  currency (ISO 4217 code such as USD, or null), po_number (string or null),
  subtotal (number or null), tax (number or null), total (number),
  line_items (list of objects with description, quantity, unit_price, amount).
The document may be in English or Spanish and may contain OCR damage; copy text as it appears."""


class ExtractionError(RuntimeError):
    """A structured extraction failure: which document, which stage, and the individual problems."""

    def __init__(self, doc_id: str, stage: str, message: str, errors: Optional[List[Dict[str, str]]] = None):
        super().__init__(f"extraction failed for {doc_id} at stage {stage!r}: {message}")
        self.doc_id = doc_id
        self.stage = stage  # "input" | "provider" | "response_shape" | "validation"
        self.message = message
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "stage": self.stage, "message": self.message, "errors": self.errors}


class ForbiddenInputError(ValueError):
    """Raised when extraction is handed anything beyond reader output (for example ground truth)."""


class ExtractedLineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    quantity: float
    unit_price: float
    amount: float

    @field_validator("description", mode="before")
    @classmethod
    def _description(cls, v: Any) -> str:
        text = optional_text(v)
        if text is None:
            raise ValueError("description is empty")
        return text

    @field_validator("quantity", mode="before")
    @classmethod
    def _quantity(cls, v: Any) -> float:
        return parse_quantity(v)

    @field_validator("unit_price", "amount", mode="before")
    @classmethod
    def _money(cls, v: Any) -> float:
        return parse_amount(v)


class ExtractedInvoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = None
    po_number: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: float
    line_items: List[ExtractedLineItem] = []

    @field_validator("vendor_name", "invoice_number", "po_number", mode="before")
    @classmethod
    def _text(cls, v: Any) -> Optional[str]:
        return optional_text(v)

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Optional[str]:
        return parse_date(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v: Any) -> Optional[str]:
        return parse_currency(v)

    @field_validator("subtotal", "tax", mode="before")
    @classmethod
    def _optional_money(cls, v: Any) -> Optional[float]:
        return None if v is None or (isinstance(v, str) and not v.strip()) else parse_amount(v)

    @field_validator("total", mode="before")
    @classmethod
    def _total(cls, v: Any) -> float:
        return parse_amount(v)


def build_extraction_prompt(reader_text: str) -> str:
    """Build the extraction prompt from the fixed instructions and the reader text, and nothing else."""
    return f"{INSTRUCTIONS}\n\n<<<DOCUMENT TEXT\n{reader_text}\nDOCUMENT TEXT>>>\n"


def _as_reader_output(reader_output: Union[ReaderOutput, Mapping[str, Any]]) -> ReaderOutput:
    """Accept a ReaderOutput (or a mapping of exactly its fields); reject anything carrying extra data."""
    if isinstance(reader_output, Mapping):
        extra = set(reader_output) - ALLOWED_INPUT_KEYS
        if extra:
            raise ForbiddenInputError(
                f"extraction input may hold only reader output fields {sorted(ALLOWED_INPUT_KEYS)}; "
                f"rejected keys: {sorted(extra)}"
            )
        return ReaderOutput(**reader_output)
    if isinstance(reader_output, ReaderOutput):
        extra = set(vars(reader_output)) - ALLOWED_INPUT_KEYS
        if extra or getattr(reader_output, "model_extra", None):
            raise ForbiddenInputError(f"reader output carries unexpected attributes: {sorted(extra)}")
        return reader_output
    raise ForbiddenInputError(f"expected ReaderOutput, got {type(reader_output).__name__}")


def extract_invoice(reader_output: Union[ReaderOutput, Mapping[str, Any]], provider: LLMProvider) -> ExtractedInvoice:
    """Extract and validate invoice fields from reader output using `provider`.

    Raises ForbiddenInputError for inputs beyond reader output, and ExtractionError for a missing or invalid
    response. NotImplementedError from a provider that is not enabled is deliberately not caught.
    """
    ro = _as_reader_output(reader_output)
    if not ro.text.strip():
        raise ExtractionError(ro.doc_id, "input", "reader text is empty")
    prompt = build_extraction_prompt(ro.text)
    try:
        raw = provider.complete(TASK, prompt, ro.doc_id)
    except LLMProviderError as exc:
        raise ExtractionError(ro.doc_id, "provider", str(exc)) from exc
    if not isinstance(raw, dict):
        raise ExtractionError(ro.doc_id, "response_shape", f"expected a JSON object, got {type(raw).__name__}")
    try:
        return ExtractedInvoice.model_validate(raw)
    except ValidationError as exc:
        errors = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
        raise ExtractionError(ro.doc_id, "validation", f"{len(errors)} invalid field(s)", errors) from exc

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class LineItem(BaseModel):
    description: str
    qty: int = 1
    unit_price: float
    amount: float
    gl_account: Optional[str] = None
    gl_account_name: Optional[str] = None
    department: Optional[str] = None

class FieldConfidence(BaseModel):
    vendor_name: float = 1.0
    invoice_number: float = 1.0
    date: float = 1.0
    total_amount: float = 1.0
    line_items: float = 1.0

class DecisionStep(BaseModel):
    step_index: int
    node_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action: str
    reasoning: str
    confidence: float = 1.0
    output_summary: Dict[str, Any] = Field(default_factory=dict)

class ValidationFlags(BaseModel):
    duplicate_found: bool = False
    po_matched: bool = True
    po_mismatch_reason: Optional[str] = None
    exceeds_auto_post_ceiling: bool = False
    low_confidence_fields: bool = False
    unknown_vendor: bool = False

class ExtractedInvoiceFields(BaseModel):
    vendor_name: Optional[str] = None
    vendor_id: Optional[str] = None
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    po_number: Optional[str] = None
    total_amount: float = 0.0
    line_items: List[LineItem] = Field(default_factory=list)

class InvoiceState(BaseModel):
    invoice_id: str
    # File stem of the document the state was read from; None for payloads that carry simulated extraction.
    # The GL-Coder node uses it to choose the LLM coder over the keyword rules.
    document_id: Optional[str] = None
    raw_text: str = ""
    extracted_fields: ExtractedInvoiceFields = Field(default_factory=ExtractedInvoiceFields)
    field_confidence: FieldConfidence = Field(default_factory=FieldConfidence)
    validation_flags: ValidationFlags = Field(default_factory=ValidationFlags)
    route_signal: str = "pending"  # "auto_post" or "human_review"
    human_decision: Optional[str] = None  # "approved" or "rejected"
    human_reasoning: Optional[str] = None
    posted_entry_id: Optional[str] = None
    posting_timestamp: Optional[str] = None
    decision_trail: List[DecisionStep] = Field(default_factory=list)

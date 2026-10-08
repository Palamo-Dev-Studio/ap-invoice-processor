import os
import json
import random
from typing import Any, AsyncGenerator
from datetime import datetime

from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.workflow import node

from ap_invoice_processor.models import InvoiceState, DecisionStep, LineItem
from ap_invoice_processor.document_intake import fill_state_from_document
from ap_invoice_processor.keyword_coder import keyword_code, match_vendor
from ap_invoice_processor.llm.gl import code_lines, load_chart
from ap_invoice_processor.llm.provider import get_provider
from ap_invoice_processor.skill_loader import load_skill_rules
from mcp_server.netsuite_mcp_client import post_invoice_sync

def _load_json_data(filename: str) -> Any:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def _parse_input_to_dict(node_input: Any) -> dict:
    if isinstance(node_input, dict):
        return node_input
    if hasattr(node_input, "parts") and node_input.parts:
        for part in node_input.parts:
            if hasattr(part, "text") and part.text:
                try:
                    return json.loads(part.text)
                except Exception:
                    pass
    if isinstance(node_input, str):
        try:
            return json.loads(node_input)
        except Exception:
            pass
    return {}

@node
def intake_node(ctx: Context, node_input: Any) -> Event:
    """Intake node: pulls raw invoice payload and initializes normalized shared state."""
    payload = _parse_input_to_dict(node_input)
    document_path = payload.get("document_path") if payload else None
    
    if payload and "id" in payload:
        invoice_id = payload.get("id")
        raw_text = payload.get("raw_text", "")
        extracted_sim = payload.get("simulated_extraction", {})
    else:
        invoice_id = f"INV-{random.randint(1000, 9999)}"
        raw_text = str(node_input)
        extracted_sim = {}

    invoice_state = InvoiceState(
        invoice_id=invoice_id,
        raw_text=raw_text
    )

    doc_summary = None
    if document_path:
        # A document payload is read and extracted through the configured LLM provider. Any simulated
        # extraction in the same payload is ignored so the two sources never mix.
        extracted_sim = {}
        doc_summary = fill_state_from_document(invoice_state, document_path)
        invoice_state.document_id = doc_summary["doc_id"]
        if not (payload and "id" in payload):
            invoice_state.invoice_id = invoice_id = doc_summary["doc_id"]

    if extracted_sim:
        for k, v in extracted_sim.items():
            if k == "line_items":
                invoice_state.extracted_fields.line_items = [LineItem(**item) for item in v]
            elif k == "confidence":
                for conf_k, conf_v in v.items():
                    setattr(invoice_state.field_confidence, conf_k, conf_v)
            elif hasattr(invoice_state.extracted_fields, k):
                setattr(invoice_state.extracted_fields, k, v)

    if doc_summary is not None:
        step = DecisionStep(
            step_index=len(invoice_state.decision_trail) + 1,
            node_name="Intake",
            action="Read Document & Extract Fields via LLM Provider",
            reasoning=f"Read document for ID {invoice_id}; extraction result: {doc_summary['extraction']}.",
            confidence=1.0 if doc_summary["extraction"] == "ok" else 0.0,
            output_summary={"invoice_id": invoice_id, "raw_length": len(invoice_state.raw_text), **doc_summary}
        )
    else:
        step = DecisionStep(
            step_index=len(invoice_state.decision_trail) + 1,
            node_name="Intake",
            action="Pull & Normalize Raw Invoice",
            reasoning=f"Successfully ingested raw invoice payload for ID {invoice_id}.",
            confidence=1.0,
            output_summary={"invoice_id": invoice_id, "raw_length": len(raw_text)}
        )
    invoice_state.decision_trail.append(step)

    state_dict = invoice_state.model_dump()
    return Event(output=state_dict, state={"invoice_state": state_dict})

@node
def extractor_node(ctx: Context, node_input: Any) -> Event:
    """Extractor node: structured extraction & confidence calculation."""
    state_dict = node_input if isinstance(node_input, dict) and "invoice_id" in node_input else ctx.state.get("invoice_state", {})
    invoice_state = InvoiceState(**state_dict)

    fields = invoice_state.extracted_fields
    conf = invoice_state.field_confidence

    avg_conf = (conf.vendor_name + conf.invoice_number + conf.date + conf.total_amount + conf.line_items) / 5.0

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="Extractor",
        action="Structured Field Extraction & Confidence Scoring",
        reasoning=f"Extracted vendor '{fields.vendor_name}', total ${fields.total_amount:.2f}, with average confidence {avg_conf:.2f}.",
        confidence=avg_conf,
        output_summary={
            "vendor_name": fields.vendor_name,
            "total_amount": fields.total_amount,
            "po_number": fields.po_number,
            "line_items_count": len(fields.line_items),
            "avg_confidence": round(avg_conf, 2)
        }
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    return Event(output=res_dict, state={"invoice_state": res_dict})

def _gl_code_document(invoice_state: InvoiceState) -> Event:
    """Code a document-intake invoice with the LLM coder; per-line failures fall back to the keyword coder.

    A provider that is not enabled (NotImplementedError) or a misconfigured provider name (ValueError)
    propagates rather than silently falling back to the keyword coder.
    """
    fields = invoice_state.extracted_fields
    provider = get_provider()
    codes = code_lines(
        fields.line_items,
        load_chart(),
        provider,
        invoice_state.document_id,
        vendor_name=fields.vendor_name,
        skill_rules=load_skill_rules(),
    )

    lines_summary = []
    for i, (item, code) in enumerate(zip(fields.line_items, codes)):
        item.gl_account = code.account
        item.gl_account_name = code.account_name
        item.department = code.department
        lines_summary.append(
            {
                "line": i,
                "account": code.account,
                "source": code.source,
                "confidence": code.confidence,
                "reason": code.reason,
            }
        )
    llm_count = sum(1 for code in codes if code.source == "llm")
    fallback_count = len(codes) - llm_count

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="GL-Coder",
        action="Code Line Items via LLM Provider with Keyword Fallback",
        reasoning=(
            f"Coded {len(codes)} line items through the {type(provider).__name__}: "
            f"{llm_count} by the LLM coder, {fallback_count} by the keyword fallback."
        ),
        confidence=min((code.confidence for code in codes), default=0.0),
        output_summary={
            "coded_line_items": len(codes),
            "llm_coded": llm_count,
            "keyword_fallback": fallback_count,
            "provider": type(provider).__name__,
            "lines": lines_summary,
        },
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    return Event(output=res_dict, state={"invoice_state": res_dict})


@node
def gl_coder_node(ctx: Context, node_input: Any) -> Event:
    """GL-Coder node: map line items to GL accounts using SKILL.md rules."""
    state_dict = node_input if isinstance(node_input, dict) and "invoice_id" in node_input else ctx.state.get("invoice_state", {})
    invoice_state = InvoiceState(**state_dict)

    if invoice_state.document_id is not None:
        return _gl_code_document(invoice_state)

    skill_rules = load_skill_rules()
    vendor_master = _load_json_data("vendor_master.json")

    matched_vendor_entry = match_vendor(invoice_state.extracted_fields.vendor_name, vendor_master)

    coded_count = 0
    for item in invoice_state.extracted_fields.line_items:
        gl, gl_name, dept, _rule = keyword_code(
            invoice_state.extracted_fields.vendor_name, item.description, matched_vendor_entry, skill_rules
        )

        item.gl_account = gl
        item.gl_account_name = gl_name
        item.department = dept
        coded_count += 1

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="GL-Coder",
        action="Apply Portable Agent Skill GL Rules",
        reasoning=f"Mapped {coded_count} line items to GL accounts based on SKILL.md vendor & keyword rules.",
        confidence=0.95 if matched_vendor_entry else 0.85,
        output_summary={"coded_line_items": coded_count, "matched_vendor": matched_vendor_entry["name"] if matched_vendor_entry else "Fallback"}
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    return Event(output=res_dict, state={"invoice_state": res_dict})

@node
def policy_validator_node(ctx: Context, node_input: Any) -> Event:
    """Policy-Validator node: check hard ceiling ($5k), duplicates, PO match, confidence."""
    state_dict = node_input if isinstance(node_input, dict) and "invoice_id" in node_input else ctx.state.get("invoice_state", {})
    invoice_state = InvoiceState(**state_dict)

    skill_rules = load_skill_rules()
    po_db = _load_json_data("po_database.json")
    historical_invoices = _load_json_data("historical_invoices.json")

    flags = invoice_state.validation_flags
    fields = invoice_state.extracted_fields
    conf = invoice_state.field_confidence

    if isinstance(historical_invoices, list) and fields.invoice_number in historical_invoices:
        flags.duplicate_found = True

    if fields.total_amount >= skill_rules.auto_post_ceiling:
        flags.exceeds_auto_post_ceiling = True

    min_extracted_conf = min(conf.vendor_name, conf.invoice_number, conf.date, conf.total_amount)
    if min_extracted_conf < skill_rules.min_confidence:
        flags.low_confidence_fields = True

    vendor_master = _load_json_data("vendor_master.json")
    matched_vendor = False
    po_req = False
    if isinstance(vendor_master, list):
        v_clean = (fields.vendor_name or "").lower()
        for vm in vendor_master:
            if vm["name"].lower() in v_clean or any(alias.lower() in v_clean for alias in vm.get("aliases", [])):
                matched_vendor = True
                po_req = vm.get("po_required", False)
                invoice_state.extracted_fields.vendor_id = vm.get("vendor_id")
                break
    if not matched_vendor:
        flags.unknown_vendor = True

    if po_req or fields.po_number:
        po = fields.po_number
        if not po or po not in po_db:
            flags.po_matched = False
            flags.po_mismatch_reason = f"PO '{po}' not found in PO database."

    needs_human = (
        flags.duplicate_found or
        flags.exceeds_auto_post_ceiling or
        flags.low_confidence_fields or
        flags.unknown_vendor or
        not flags.po_matched
    )

    route_signal = "human_review" if needs_human else "auto_post"
    invoice_state.route_signal = route_signal

    flag_list = []
    if flags.duplicate_found: flag_list.append("Duplicate Invoice")
    if flags.exceeds_auto_post_ceiling: flag_list.append(f"Exceeds ${skill_rules.auto_post_ceiling:.0f} Ceiling")
    if flags.low_confidence_fields: flag_list.append("Low Confidence")
    if flags.unknown_vendor: flag_list.append("Unknown Vendor")
    if not flags.po_matched: flag_list.append("PO Mismatch")

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="Policy-Validator",
        action="Evaluate Compliance & Hard Safety Rails",
        reasoning=f"Validation evaluated. Route = '{route_signal}'. Tripped flags: {', '.join(flag_list) if flag_list else 'None'}.",
        confidence=1.0,
        output_summary={"route_signal": route_signal, "flags": flag_list}
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    return Event(output=res_dict, route=route_signal, state={"invoice_state": res_dict})

@node(rerun_on_resume=True)
async def human_gate_node(ctx: Context, node_input: Any) -> AsyncGenerator[Any, None]:
    """Human Gate node: pauses for user triage approval/rejection if flagged."""
    state_dict = node_input if isinstance(node_input, dict) and "invoice_id" in node_input else ctx.state.get("invoice_state", {})
    invoice_state = InvoiceState(**state_dict)

    decision = invoice_state.human_decision
    reasoning = invoice_state.human_reasoning

    if not decision and getattr(ctx, "resume_inputs", None) and "human_triage" in ctx.resume_inputs:
        triage_data = ctx.resume_inputs["human_triage"]
        if isinstance(triage_data, dict):
            decision = triage_data.get("decision", "approved").lower()
            reasoning = triage_data.get("reasoning", "Reviewed via triage portal.")
        elif isinstance(triage_data, str):
            decision = triage_data.lower()
            reasoning = "Reviewed via triage portal."

    if not decision:
        yield RequestInput(
            interrupt_id="human_triage",
            message=f"HUMAN GATE PAUSE: Invoice {invoice_state.invoice_id} (${invoice_state.extracted_fields.total_amount:.2f}) requires human review."
        )
        return

    invoice_state.human_decision = decision
    invoice_state.human_reasoning = reasoning or "Approved by reviewer."

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="Human Gate",
        action="Interactive Human-in-the-Loop Triage",
        reasoning=f"Human reviewer submitted decision: '{decision.upper()}'. Reasoning: {invoice_state.human_reasoning}",
        confidence=1.0,
        output_summary={"decision": decision, "reasoning": invoice_state.human_reasoning}
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    yield Event(output=res_dict, state={"invoice_state": res_dict})

@node
def poster_node(ctx: Context, node_input: Any) -> Event:
    """Poster node: executes NetSuite ERP GL posting for approved entries."""
    state_dict = node_input if isinstance(node_input, dict) and "invoice_id" in node_input else ctx.state.get("invoice_state", {})
    invoice_state = InvoiceState(**state_dict)

    if invoice_state.human_decision == "rejected":
        step = DecisionStep(
            step_index=len(invoice_state.decision_trail) + 1,
            node_name="Poster",
            action="Abort NetSuite GL Posting",
            reasoning="Invoice posting aborted because human reviewer rejected the entry.",
            confidence=1.0,
            output_summary={"status": "aborted", "reason": invoice_state.human_reasoning}
        )
        invoice_state.decision_trail.append(step)
        res_dict = invoice_state.model_dump()
        return Event(output=res_dict, state={"invoice_state": res_dict})

    # Post through the NetSuite MCP server: this node acts as an MCP client, spawns the
    # mock server over stdio, and calls its post_invoice_to_netsuite tool to obtain the
    # transaction id. No real ERP / network - the server returns synthetic mock data.
    first_line = invoice_state.extracted_fields.line_items[0] if invoice_state.extracted_fields.line_items else None
    gl_account = first_line.gl_account if first_line and first_line.gl_account else "6100"

    mcp_result = post_invoice_sync(
        invoice_id=invoice_state.invoice_id,
        vendor=invoice_state.extracted_fields.vendor_name or "Unknown Vendor",
        amount=invoice_state.extracted_fields.total_amount,
        gl_account=gl_account,
    )

    ns_id = mcp_result["transaction_id"]
    timestamp = mcp_result.get("timestamp") or datetime.now().isoformat()
    invoice_state.posted_entry_id = ns_id
    invoice_state.posting_timestamp = timestamp

    step = DecisionStep(
        step_index=len(invoice_state.decision_trail) + 1,
        node_name="Poster",
        action="Post GL Entry via NetSuite MCP Server",
        reasoning=f"Posted GL entry via the NetSuite MCP server (post_invoice_to_netsuite tool). Transaction ID '{ns_id}', status '{mcp_result.get('status', 'posted')}'.",
        confidence=1.0,
        output_summary={
            "netsuite_transaction_id": ns_id,
            "mcp_status": mcp_result.get("status", "posted"),
            "posted_timestamp": timestamp,
            "total_posted_amount": invoice_state.extracted_fields.total_amount,
            "posted_via": "netsuite_mcp_server"
        }
    )
    invoice_state.decision_trail.append(step)

    res_dict = invoice_state.model_dump()
    return Event(output=res_dict, state={"invoice_state": res_dict})

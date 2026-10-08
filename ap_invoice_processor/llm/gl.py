# ABOUTME: LLM GL coder behind the LLMProvider interface; falls back to the keyword coder on failure or an off-chart account.
# ABOUTME: Prompts carry only the chart and line descriptions; confidences on fallback codes are rule-strength heuristics.
import json
import os
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel

from ap_invoice_processor.keyword_coder import keyword_code, match_vendor
from ap_invoice_processor.llm.provider import LLMProvider, LLMProviderError
from ap_invoice_processor.skill_loader import SkillRules, load_skill_rules

TASK = "gl"
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CHART_PATH = os.path.join(_DATA_DIR, "gl_chart_of_accounts.json")
VENDOR_MASTER_PATH = os.path.join(_DATA_DIR, "vendor_master.json")
MAX_REASON_CHARS = 300

# Confidence reported for a keyword-coder result, by the rule that produced it. These describe how specific the
# rule is, not how often it is right.
KEYWORD_CONFIDENCE = {"vendor_master": 0.7, "vendor_keyword": 0.7, "description_keyword": 0.6, "default": 0.3}

INSTRUCTIONS = """You assign a general-ledger account to each invoice line item.
Choose only from the chart of accounts below. Pick the account whose name best covers the expense the line
describes; if no account fits well, still choose the closest one and give it a low confidence.
Return a single JSON object: {"lines": [{"line": <index>, "account": "<account number>",
"confidence": <number from 0 to 1>, "reason": "<one short sentence>"}]}, with one entry per line, in order."""


class GLCode(BaseModel):
    account: str
    account_name: str
    department: Optional[str] = None
    confidence: float
    reason: str
    source: Literal["llm", "keyword_fallback"]


def load_chart(path: str = CHART_PATH) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_vendor_master() -> Any:
    try:
        with open(VENDOR_MASTER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return []


def _description(line: Any) -> str:
    if isinstance(line, str):
        return line
    if isinstance(line, Mapping):
        return str(line.get("description", ""))
    return str(getattr(line, "description", ""))


def build_gl_prompt(descriptions: Sequence[str], chart: Sequence[Mapping[str, Any]]) -> str:
    """Build the coding prompt from the fixed instructions, the chart and the line descriptions only."""
    chart_lines = "\n".join(
        f"  {a['account_number']}  {a['account_name']}  (category: {a.get('category', '')})" for a in chart
    )
    line_block = "\n".join(f"  {i}: {d}" for i, d in enumerate(descriptions))
    return f"{INSTRUCTIONS}\n\nCHART OF ACCOUNTS\n{chart_lines}\n\nLINE ITEMS\n{line_block}\n"


def _fallback(
    why: str,
    vendor_name: Optional[str],
    description: str,
    chart_by_number: Mapping[str, Mapping[str, Any]],
    matched_vendor: Optional[Dict[str, Any]],
    skill_rules: SkillRules,
) -> GLCode:
    result = keyword_code(vendor_name, description, matched_vendor, skill_rules)
    entry = chart_by_number.get(result.gl)
    return GLCode(
        account=result.gl,
        account_name=entry["account_name"] if entry else (result.gl_name or ""),
        department=result.department,
        confidence=KEYWORD_CONFIDENCE[result.rule],
        reason=f"keyword fallback ({result.rule}): {why}",
        source="keyword_fallback",
    )


def _valid_entry(entry: Any, chart_by_number: Mapping[str, Any]) -> Optional[str]:
    """Return None when the response entry is usable, else a short reason it is not."""
    if not isinstance(entry, Mapping):
        return "response entry is not an object"
    account = entry.get("account")
    if not isinstance(account, str) or account not in chart_by_number:
        return f"account {account!r} is not in the chart"
    confidence = entry.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return f"confidence {confidence!r} is not a number between 0 and 1"
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "reason is missing"
    return None


def code_lines(
    lines: Sequence[Any],
    chart: Sequence[Mapping[str, Any]],
    provider: LLMProvider,
    doc_id: str,
    *,
    vendor_name: Optional[str] = None,
    skill_rules: Optional[SkillRules] = None,
) -> List[GLCode]:
    """Code each line with the provider; use the keyword coder for any line the provider cannot code validly.

    `vendor_name` is used only by the keyword fallback (the prompt never carries it). A provider that raises
    LLMProviderError, or returns an unusable response, sends every line to the fallback; an individual entry
    that is missing, off-chart or malformed sends just that line. NotImplementedError from a provider that is
    not enabled is not caught.
    """
    descriptions = [_description(line) for line in lines]
    if not descriptions:
        return []
    chart_by_number = {a["account_number"]: a for a in chart}
    rules = skill_rules or load_skill_rules()
    matched_vendor = match_vendor(vendor_name, _load_vendor_master())

    def fallback(i: int, why: str) -> GLCode:
        return _fallback(why, vendor_name, descriptions[i], chart_by_number, matched_vendor, rules)

    try:
        raw = provider.complete(TASK, build_gl_prompt(descriptions, chart), doc_id)
    except LLMProviderError as exc:
        return [fallback(i, f"provider failed ({exc})") for i in range(len(descriptions))]
    entries = raw.get("lines") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return [fallback(i, "provider response has no 'lines' list") for i in range(len(descriptions))]
    by_index = {e["line"]: e for e in entries if isinstance(e, Mapping) and isinstance(e.get("line"), int)}

    codes: List[GLCode] = []
    for i in range(len(descriptions)):
        entry = by_index.get(i)
        problem = "no entry for this line" if entry is None else _valid_entry(entry, chart_by_number)
        if problem:
            codes.append(fallback(i, problem))
            continue
        account = chart_by_number[entry["account"]]
        codes.append(
            GLCode(
                account=entry["account"],
                account_name=account["account_name"],
                department=account.get("default_department"),
                confidence=float(entry["confidence"]),
                reason=entry["reason"].strip()[:MAX_REASON_CHARS],
                source="llm",
            )
        )
    return codes

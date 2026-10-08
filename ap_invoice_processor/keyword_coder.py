# ABOUTME: Keyword/rule GL coder shared by the GL-Coder node and the LLM coder's fallback path.
# ABOUTME: Applies vendor-master default, SKILL.md vendor keywords, then description keywords, then a 6100 default.
from typing import Any, Dict, List, NamedTuple, Optional

from ap_invoice_processor.skill_loader import SkillRules

DEFAULT_GL = "6100"
DEFAULT_GL_NAME = "Office Supplies & Software (Fallback)"
DEFAULT_DEPARTMENT = "Administration"


class KeywordResult(NamedTuple):
    gl: str
    gl_name: Optional[str]
    department: Optional[str]
    # Which rule produced the code: "vendor_master" | "vendor_keyword" | "description_keyword" | "default".
    rule: str


def match_vendor(vendor_name: Optional[str], vendor_master: Any) -> Optional[Dict[str, Any]]:
    """Return the first vendor-master entry whose name or alias occurs in the vendor name, else None."""
    vendor_name_clean = (vendor_name or "").lower()
    if isinstance(vendor_master, list):
        for vm in vendor_master:
            if vm["name"].lower() in vendor_name_clean or any(
                alias.lower() in vendor_name_clean for alias in vm.get("aliases", [])
            ):
                return vm
    return None


def keyword_code(
    vendor_name: Optional[str],
    description: str,
    matched_vendor_entry: Optional[Dict[str, Any]],
    skill_rules: SkillRules,
) -> KeywordResult:
    """Code one line item with the keyword rules, in the order the GL-Coder node has always applied them."""
    vendor_name_clean = (vendor_name or "").lower()
    gl = None
    gl_name = None
    dept = None
    rule = ""

    if matched_vendor_entry:
        gl = matched_vendor_entry.get("default_gl_account")
        dept = matched_vendor_entry.get("default_department")
        if gl:
            rule = "vendor_master"

    if not gl:
        for vendor_rule in skill_rules.vendor_mappings:
            if any(kw in vendor_name_clean for kw in vendor_rule["keywords"]):
                gl = vendor_rule["gl"]
                gl_name = vendor_rule["gl_name"]
                dept = vendor_rule["department"]
                rule = "vendor_keyword"
                break

    if not gl:
        desc_clean = description.lower()
        for fb in skill_rules.fallback_keywords:
            if any(kw in desc_clean for kw in fb["keywords"]):
                gl = fb["gl"]
                gl_name = fb["gl_name"]
                dept = fb["department"]
                rule = "description_keyword"
                break

    if not gl:
        gl = DEFAULT_GL
        gl_name = DEFAULT_GL_NAME
        dept = DEFAULT_DEPARTMENT
        rule = "default"

    return KeywordResult(gl=gl, gl_name=gl_name, department=dept, rule=rule)

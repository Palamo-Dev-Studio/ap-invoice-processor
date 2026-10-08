# ABOUTME: Characterization tests pinning the extracted keyword GL coder to the original inline GL-Coder node logic.
# ABOUTME: The oracle below is a verbatim copy of that logic as it stood before the extraction.
import glob
import json
import os

import pytest

from ap_invoice_processor.keyword_coder import keyword_code, match_vendor
from ap_invoice_processor.skill_loader import load_skill_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "vendor_master.json"), encoding="utf-8") as _f:
    VENDOR_MASTER = json.load(_f)
RULES = load_skill_rules()


def _legacy(vendor_name, description):
    """The pre-extraction GL-Coder node logic for one line item, returning (gl, gl_name, dept)."""
    vendor_name_clean = (vendor_name or "").lower()
    matched_vendor_entry = None
    if isinstance(VENDOR_MASTER, list):
        for vm in VENDOR_MASTER:
            if vm["name"].lower() in vendor_name_clean or any(alias.lower() in vendor_name_clean for alias in vm.get("aliases", [])):
                matched_vendor_entry = vm
                break
    gl = None
    gl_name = None
    dept = None
    if matched_vendor_entry:
        gl = matched_vendor_entry.get("default_gl_account")
        dept = matched_vendor_entry.get("default_department")
    if not gl:
        for rule in RULES.vendor_mappings:
            if any(kw in vendor_name_clean for kw in rule["keywords"]):
                gl = rule["gl"]
                gl_name = rule["gl_name"]
                dept = rule["department"]
                break
    if not gl:
        desc_clean = description.lower()
        for fb in RULES.fallback_keywords:
            if any(kw in desc_clean for kw in fb["keywords"]):
                gl = fb["gl"]
                gl_name = fb["gl_name"]
                dept = fb["department"]
                break
    if not gl:
        gl = "6100"
        gl_name = "Office Supplies & Software (Fallback)"
        dept = "Administration"
    return gl, gl_name, dept


def _corpus_cases():
    cases = set()
    for path in glob.glob(os.path.join(ROOT, "data", "corpus", "ground_truth", "*.json")):
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        for li in gt["line_items"]:
            cases.add((gt["vendor_name"], li["description"]))
    return sorted(cases)


HAND_CASES = [
    ("Amazon Web Services", "Cloud EC2"),
    ("Apple Hardware Direct", "MacBook Pro"),
    ("Unknown Vendor LLC", "Dedicated hosting"),
    ("Unknown Vendor LLC", "Legal audit"),
    ("Unknown Vendor LLC", "Mystery item"),
    (None, "Mystery item"),
    ("", "Standing desk"),
]


@pytest.mark.parametrize("vendor,description", HAND_CASES + _corpus_cases())
def test_keyword_code_matches_legacy_logic(vendor, description):
    matched = match_vendor(vendor, VENDOR_MASTER)
    result = keyword_code(vendor, description, matched, RULES)
    assert (result.gl, result.gl_name, result.department) == _legacy(vendor, description)


def test_rule_labels_identify_the_path_taken():
    aws = match_vendor("Amazon Web Services", VENDOR_MASTER)
    assert keyword_code("Amazon Web Services", "x", aws, RULES).rule == "vendor_master"
    assert keyword_code("Acme Cloud Ads", "x", None, RULES).rule == "vendor_keyword"
    assert keyword_code("Nobody Inc", "server rack", None, RULES).rule == "description_keyword"
    assert keyword_code("Nobody Inc", "zzz", None, RULES).rule == "default"


def test_match_vendor_handles_non_list_master_and_missing_name():
    assert match_vendor("Amazon Web Services", {}) is None
    assert match_vendor(None, VENDOR_MASTER) is None

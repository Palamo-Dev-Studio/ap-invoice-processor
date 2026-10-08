# ABOUTME: Dashboard presentation helpers: demo-mode page rendering and node subtitles that name the coder that actually ran.
# ABOUTME: Pure functions over strings and state dicts, so the page behaviour is testable without a browser.
import os
import re
from typing import Any, Dict, Mapping, Optional

DEMO_MODE_ENV = "AP_DEMO_MODE"
DEMO_NOTICE = "Test build · synthetic data only · results are not financial advice"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
# index.html brackets the cost/ROI banner with these markers so demo mode can drop it from the served page entirely.
_ROI_BANNER = re.compile(r"<!--roi-banner:start-->.*?<!--roi-banner:end-->", re.DOTALL)
_NOTICE_SLOT = "<!--demo-notice-->"


def demo_mode_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return (env.get(DEMO_MODE_ENV) or "").strip().lower() in _TRUTHY


def render_index(html: str, demo_mode: bool) -> str:
    """Return the dashboard page. In demo mode the cost/ROI banner is removed and the fixed test notice is shown."""
    if demo_mode:
        html = _ROI_BANNER.sub("", html)
        return html.replace(_NOTICE_SLOT, f'<div class="demo-notice" role="note">{DEMO_NOTICE}</div>')
    return html.replace(_NOTICE_SLOT, "")


def _step_summary(state: Mapping[str, Any], node_name: str) -> Optional[Dict[str, Any]]:
    for step in state.get("decision_trail") or []:
        if step.get("node_name") == node_name:
            return step.get("output_summary") or {}
    return None


def node_subtitles(state: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Subtitles for the pipeline nodes whose step has run, taken from what the decision trail records.

    A document run extracts inside Intake and codes with the provider named in the GL-Coder step; a simulated-extraction
    run reads pre-structured fields and codes with the SKILL.md keyword rules. Nodes that have not run are omitted, so
    the page keeps its neutral default for them.
    """
    if not state:
        return {}
    labels: Dict[str, str] = {}
    intake = _step_summary(state, "Intake")
    is_document = state.get("document_id") is not None
    if intake is not None:
        labels["Intake"] = "Read document + extract" if is_document else "Raw -> State"
    if _step_summary(state, "Extractor") is not None:
        labels["Extractor"] = "Confidence scoring" if is_document else "Pre-structured fields"
    gl = _step_summary(state, "GL-Coder")
    if gl is not None:
        provider = gl.get("provider")
        if provider == "AnthropicProvider":
            labels["GL-Coder"] = "LLM coder + keyword fallback"
        elif provider == "FixtureProvider":
            labels["GL-Coder"] = "Fixture coder + keyword fallback"
        elif provider:
            labels["GL-Coder"] = f"{provider} + keyword fallback"
        else:
            labels["GL-Coder"] = "SKILL.md keyword rules"
    return labels

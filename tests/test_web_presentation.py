# ABOUTME: Tests for web/presentation.py: demo-mode page rendering and the node subtitles derived from the decision trail.
# ABOUTME: Subtitles must name the coder that actually ran, so each provider and the simulated path are pinned separately.
import pytest

from web.presentation import DEMO_NOTICE, demo_mode_enabled, node_subtitles, render_index

PAGE = "<body><!--demo-notice--><!--roi-banner:start--><div>$14.50 87.9%</div><!--roi-banner:end--><main>x</main></body>"


def _state(*steps, document_id=None):
    return {"document_id": document_id, "decision_trail": [{"node_name": n, "output_summary": s} for n, s in steps]}


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False), ("off", False), (None, False)])
def test_demo_mode_flag_parsing(value, expected):
    env = {} if value is None else {"AP_DEMO_MODE": value}
    assert demo_mode_enabled(env) is expected


def test_demo_rendering_drops_the_banner_and_inserts_the_notice():
    html = render_index(PAGE, True)
    assert "$14.50" not in html and "87.9%" not in html and "roi-banner" not in html and "<!--demo-notice-->" not in html
    assert DEMO_NOTICE in html and "<main>x</main>" in html


def test_normal_rendering_keeps_the_banner_and_drops_only_the_notice_slot():
    html = render_index(PAGE, False)
    assert "$14.50" in html and DEMO_NOTICE not in html and "<!--demo-notice-->" not in html


def test_a_page_with_two_banner_blocks_loses_both_and_nothing_between_them():
    html = render_index("a<!--roi-banner:start-->X<!--roi-banner:end-->b<!--roi-banner:start-->Y<!--roi-banner:end-->c", True)
    assert html == "abc"


@pytest.mark.parametrize(
    "provider,label",
    [
        ("AnthropicProvider", "LLM coder + keyword fallback"),
        ("FixtureProvider", "Fixture coder + keyword fallback"),
        ("SomethingElse", "SomethingElse + keyword fallback"),
    ],
)
def test_the_gl_coder_subtitle_names_the_provider_that_ran(provider, label):
    state = _state(("Intake", {}), ("Extractor", {}), ("GL-Coder", {"provider": provider}), document_id="en-001")
    assert node_subtitles(state)["GL-Coder"] == label


def test_the_simulated_path_names_the_skill_rules_and_the_supplied_fields():
    state = _state(("Intake", {}), ("Extractor", {}), ("GL-Coder", {"coded_line_items": 1}))
    labels = node_subtitles(state)
    assert labels == {"Intake": "Raw -> State", "Extractor": "Pre-structured fields", "GL-Coder": "SKILL.md keyword rules"}


def test_the_document_path_says_extraction_happens_in_intake():
    labels = node_subtitles(_state(("Intake", {}), ("Extractor", {}), document_id="en-001"))
    assert labels == {"Intake": "Read document + extract", "Extractor": "Confidence scoring"}


def test_nodes_that_have_not_run_get_no_subtitle():
    assert node_subtitles(_state(("Intake", {}))) == {"Intake": "Raw -> State"}
    assert node_subtitles(None) == {} and node_subtitles({}) == {}

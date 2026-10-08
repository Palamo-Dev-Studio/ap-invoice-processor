# ABOUTME: Tests for AP_SPEND_LEDGER_PATH: where the live provider's spend ledger lives, and that the per-instance cap still stops calls.
# ABOUTME: Cloud Run's filesystem is ephemeral, so the ledger path and cap must be settable from the environment alone.
import os
from decimal import Decimal

import pytest

from ap_invoice_processor.llm import anthropic_provider as ap
from ap_invoice_processor.llm.provider import LLMProviderError
from ap_invoice_processor.llm.spend import SpendCapExceeded

KEY = "sk-ant-LEDGER-TEST-not-a-real-key"
BASE_ENV = {"ANTHROPIC_API_KEY": KEY, "AP_LLM_SPEND_CAP_USD": "2"}


def _build(environ, **kwargs):
    return ap.build_anthropic_provider(environ=environ, env_file=os.devnull, **kwargs)


def test_the_ledger_path_comes_from_the_environment(tmp_path):
    path = str(tmp_path / "ledger" / "spend.json")
    provider = _build({**BASE_ENV, "AP_SPEND_LEDGER_PATH": path})
    assert provider.tracker.path == path


def test_the_ledger_argument_beats_the_environment(tmp_path):
    provider = _build({**BASE_ENV, "AP_SPEND_LEDGER_PATH": str(tmp_path / "env.json")}, ledger_path=str(tmp_path / "arg.json"))
    assert provider.tracker.path == str(tmp_path / "arg.json")


def test_without_either_the_default_ledger_is_used(tmp_path):
    provider = _build(BASE_ENV)
    assert provider.tracker.path == ap.DEFAULT_LEDGER_PATH


def test_a_home_relative_ledger_path_is_expanded(tmp_path):
    provider = _build({**BASE_ENV, "AP_SPEND_LEDGER_PATH": "~/ap-ledger/spend.json"})
    assert provider.tracker.path == os.path.expanduser("~/ap-ledger/spend.json")


def test_a_blank_ledger_setting_falls_back_to_the_default():
    provider = _build({**BASE_ENV, "AP_SPEND_LEDGER_PATH": "   "})
    assert provider.tracker.path == ap.DEFAULT_LEDGER_PATH


def test_the_environment_cap_stops_calls_in_a_fresh_ledger_directory(tmp_path):
    """A cold Cloud Run instance starts with an empty ledger; the per-instance cap alone must still refuse a call."""
    env = {**BASE_ENV, "AP_LLM_SPEND_CAP_USD": "0.0000001", "AP_SPEND_LEDGER_PATH": str(tmp_path / "fresh" / "spend.json")}
    provider = _build(env)
    with pytest.raises(SpendCapExceeded):
        provider.complete("extract", "some invoice text", "doc-1")
    assert provider.cap_reached is True
    assert provider.tracker.cap_usd == Decimal("0.0000001")


def test_the_ledger_directory_is_created_on_first_use(tmp_path):
    path = tmp_path / "does" / "not" / "exist" / "spend.json"
    provider = _build({**BASE_ENV, "AP_LLM_SPEND_CAP_USD": "0.0000001", "AP_SPEND_LEDGER_PATH": str(path)})
    with pytest.raises(LLMProviderError):
        provider.complete("extract", "text", "doc-1")
    assert path.parent.is_dir()

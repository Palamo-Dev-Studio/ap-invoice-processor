# ABOUTME: Tests for the live-LLM spend counter: pricing arithmetic, the pre-call cap check and ledger persistence.
# ABOUTME: Pure arithmetic and temp-file tests; no network and no SDK client is created.
import json
import os
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ap_invoice_processor.llm import spend
from ap_invoice_processor.llm.provider import LLMProviderError
from ap_invoice_processor.llm.spend import (
    SpendCapExceeded,
    SpendLedgerError,
    SpendTracker,
    cost_usd,
    estimate_cost_usd,
)

MODEL = "claude-haiku-5-5"
D = Decimal


def usage(inp=0, out=0, cache_read=0, cache_write=0, write_5m=None, write_1h=None):
    cache_creation = None
    if write_5m is not None or write_1h is not None:
        cache_creation = SimpleNamespace(ephemeral_5m_input_tokens=write_5m or 0, ephemeral_1h_input_tokens=write_1h or 0)
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        cache_creation=cache_creation,
    )


# --- pricing arithmetic ----------------------------------------------------------------------------------------------
# Rates under test are the Claude Haiku 5.5 rows of https://platform.claude.com/docs/en/about-claude/pricing
# (fetched 2026-10-07): per MTok, prompts up to 100,000 tokens / over 100,000 tokens.


def test_the_standard_tier_input_and_output_rates():
    # 100,000 tokens is the largest prompt still on the standard tier; output never moves the tier.
    assert cost_usd(MODEL, usage(inp=100_000)) == D("0.01")
    assert cost_usd(MODEL, usage(out=1_000_000)) == D("0.50")


def test_a_small_call_is_priced_exactly_without_float_error():
    # 1,234 in x $0.10/M + 567 out x $0.50/M
    assert cost_usd(MODEL, usage(inp=1234, out=567)) == D("0.0001234") + D("0.0002835")


def test_cache_read_and_5m_cache_write_rates_on_the_standard_tier():
    assert cost_usd(MODEL, usage(cache_read=100_000)) == D("0.001")
    assert cost_usd(MODEL, usage(cache_write=100_000)) == D("0.0125")


def test_one_hour_cache_writes_are_priced_from_the_ttl_breakdown():
    assert cost_usd(MODEL, usage(cache_write=100_000, write_1h=100_000)) == D("0.02")
    assert cost_usd(MODEL, usage(cache_write=100_000, write_5m=100_000)) == D("0.0125")
    mixed = usage(cache_write=90_000, write_5m=30_000, write_1h=60_000)
    assert cost_usd(MODEL, mixed) == D("0.00375") + D("0.012")


def test_cache_write_tokens_the_breakdown_does_not_account_for_are_priced_at_the_dearer_one_hour_rate():
    assert cost_usd(MODEL, usage(cache_write=80_000, write_5m=40_000)) == D("0.005") + D("0.008")


def test_the_long_prompt_tier_applies_above_100k_prompt_tokens_to_every_component():
    long_usage = usage(inp=200_000, out=10_000, cache_read=50_000, cache_write=20_000)
    expected = (D(200_000) * D("0.50") + D(10_000) * D("2.50") + D(50_000) * D("0.05") + D(20_000) * D("0.625")) / D(
        1_000_000
    )
    assert cost_usd(MODEL, long_usage) == expected


def test_the_tier_boundary_is_100_000_prompt_tokens_inclusive_and_counts_cache_tokens():
    at_limit = cost_usd(MODEL, usage(inp=100_000))
    over = cost_usd(MODEL, usage(inp=100_001))
    assert at_limit == D(100_000) * D("0.10") / D(1_000_000)
    assert over == D(100_001) * D("0.50") / D(1_000_000)
    # 60k fresh + 40k cached + 1 written is a 100,001-token prompt: the long tier.
    split = cost_usd(MODEL, usage(inp=60_000, cache_read=40_000, cache_write=1))
    assert split == (D(60_000) * D("0.50") + D(40_000) * D("0.05") + D(1) * D("0.625")) / D(1_000_000)


def test_output_tokens_do_not_move_the_tier():
    assert cost_usd(MODEL, usage(inp=1_000, out=500_000)) == (D(1_000) * D("0.10") + D(500_000) * D("0.50")) / D(1_000_000)


def test_missing_optional_usage_fields_count_as_zero():
    bare = SimpleNamespace(input_tokens=10, output_tokens=20)
    assert cost_usd(MODEL, bare) == (D(10) * D("0.10") + D(20) * D("0.50")) / D(1_000_000)
    none_fields = SimpleNamespace(
        input_tokens=10, output_tokens=20, cache_read_input_tokens=None, cache_creation_input_tokens=None, cache_creation=None
    )
    assert cost_usd(MODEL, none_fields) == cost_usd(MODEL, bare)


def test_a_mapping_usage_is_accepted_like_the_sdk_object():
    assert cost_usd(MODEL, {"input_tokens": 1000, "output_tokens": 1000}) == cost_usd(MODEL, usage(inp=1000, out=1000))


def test_the_sdk_usage_model_has_the_field_names_the_counter_reads():
    from anthropic.types import Usage

    sdk = Usage(input_tokens=1000, output_tokens=1000, cache_read_input_tokens=2000, cache_creation_input_tokens=3000)
    assert cost_usd(MODEL, sdk) == (
        D(1000) * D("0.10") + D(1000) * D("0.50") + D(2000) * D("0.01") + D(3000) * D("0.125")
    ) / D(1_000_000)


def test_an_unpriced_model_is_refused_rather_than_priced_at_a_guess():
    with pytest.raises(SpendLedgerError, match="no price"):
        cost_usd("claude-unknown", usage(inp=1))
    with pytest.raises(SpendLedgerError, match="no price"):
        estimate_cost_usd("claude-unknown", "text", 100)


@pytest.mark.parametrize("bad", [-1, 1.5, "7", True])
def test_malformed_token_counts_are_refused(bad):
    with pytest.raises(SpendLedgerError, match="token count"):
        cost_usd(MODEL, usage(inp=bad))


# --- the pre-call estimate -------------------------------------------------------------------------------------------


def test_the_estimate_prices_the_output_ceiling_and_at_least_one_input_token_per_three_bytes():
    text = "x" * 3000
    est = estimate_cost_usd(MODEL, text, max_output_tokens=8000)
    floor = D(8000) * D("0.50") / D(1_000_000) + D(1000) * D("0.10") / D(1_000_000)
    assert est >= floor
    # but it is not wildly padded: well under one cent for a one-page prompt
    assert est < D("0.01")


def test_the_estimate_counts_bytes_so_cjk_text_is_not_underestimated():
    latin = estimate_cost_usd(MODEL, "a" * 3000, max_output_tokens=0)
    cjk = estimate_cost_usd(MODEL, "票" * 1000, max_output_tokens=0)  # 3,000 bytes
    assert cjk == latin


def test_the_estimate_uses_the_long_tier_for_a_very_long_prompt():
    est = estimate_cost_usd(MODEL, "x" * 400_000, max_output_tokens=1000)
    assert est >= D(200_000) * D("0.50") / D(1_000_000) + D(1000) * D("2.50") / D(1_000_000)


def test_the_estimate_covers_the_extra_text_the_request_adds_to_the_prompt():
    base = estimate_cost_usd(MODEL, "x" * 1000, max_output_tokens=0)
    with_extra = estimate_cost_usd(MODEL, "x" * 1000, max_output_tokens=0, extra_input_chars=4000)
    assert with_extra > base


# --- the tracker: ledger, cap and run cap ------------------------------------------------------------------------------


def make(tmp_path, cap="10", run_cap=None, name="spend.json"):
    return SpendTracker(str(tmp_path / name), cap_usd=D(cap), run_cap_usd=None if run_cap is None else D(run_cap))


def test_a_new_ledger_starts_at_zero_and_creates_no_file_until_something_is_recorded(tmp_path):
    t = make(tmp_path)
    assert t.total_usd == 0 and t.run_spent_usd == 0
    assert not os.path.exists(t.path)


def test_recording_adds_to_the_total_and_writes_the_ledger(tmp_path):
    t = make(tmp_path)
    assert t.record(D("0.25")) == D("0.25")
    assert t.record(D("0.5")) == D("0.75")
    with open(t.path, encoding="utf-8") as f:
        data = json.load(f)
    assert D(data["total_usd"]) == D("0.75")
    assert data["calls"] == 2
    assert t.total_usd == D("0.75") and t.run_spent_usd == D("0.75")


def test_the_total_persists_across_runs_and_the_run_counter_does_not(tmp_path):
    first = make(tmp_path)
    first.record(D("1.5"))
    second = make(tmp_path)
    assert second.total_usd == D("1.5")
    assert second.run_spent_usd == 0
    second.record(D("0.5"))
    third = make(tmp_path)
    assert third.total_usd == D("2.0")


def test_the_tracker_counts_this_runs_calls_separately_from_the_ledgers(tmp_path):
    first = make(tmp_path)
    first.record(D("0.1"))
    first.record(D("0.1"))
    second = make(tmp_path)
    second.record(D("0.1"))
    assert (first.run_calls, second.run_calls) == (2, 1)
    with open(second.path, encoding="utf-8") as f:
        assert json.load(f)["calls"] == 3


def test_many_small_records_do_not_drift(tmp_path):
    t = make(tmp_path)
    for _ in range(1000):
        t.record(D("0.000035"))
    assert make(tmp_path).total_usd == D("0.035")


def test_a_call_that_stays_under_the_cap_is_allowed_and_the_cap_itself_is_reachable(tmp_path):
    t = make(tmp_path, cap="10")
    t.record(D("9.5"))
    t.check(D("0.4"))
    t.check(D("0.5"))  # total + estimate == cap does not exceed it


def test_a_call_that_would_pass_the_cap_is_refused_before_anything_is_recorded(tmp_path):
    t = make(tmp_path, cap="10")
    t.record(D("9.5"))
    with pytest.raises(SpendCapExceeded, match="cap"):
        t.check(D("0.5001"))
    assert t.total_usd == D("9.5")


def test_the_cap_error_is_a_provider_error_so_the_pipeline_routes_it_to_human_review():
    assert issubclass(SpendCapExceeded, LLMProviderError)
    assert issubclass(SpendLedgerError, LLMProviderError)


def test_once_the_cap_is_reached_even_a_zero_cost_estimate_over_the_cap_is_refused(tmp_path):
    t = make(tmp_path, cap="1")
    t.record(D("1.2"))
    with pytest.raises(SpendCapExceeded):
        t.check(D("0"))


def test_the_total_from_earlier_runs_counts_against_the_cap(tmp_path):
    make(tmp_path, cap="10").record(D("9.9"))
    t = make(tmp_path, cap="10")
    with pytest.raises(SpendCapExceeded):
        t.check(D("0.2"))


def test_the_run_cap_limits_this_runs_spend_even_when_the_global_cap_is_far_away(tmp_path):
    make(tmp_path, cap="10").record(D("3"))
    t = make(tmp_path, cap="10", run_cap="0.5")
    t.check(D("0.5"))
    t.record(D("0.3"))
    t.check(D("0.2"))
    with pytest.raises(SpendCapExceeded, match="run cap"):
        t.check(D("0.2001"))


def test_the_global_cap_still_applies_when_the_run_cap_is_not_reached(tmp_path):
    make(tmp_path, cap="10").record(D("9.9"))
    t = make(tmp_path, cap="10", run_cap="5")
    with pytest.raises(SpendCapExceeded, match="spend cap"):
        t.check(D("0.2"))


def test_a_corrupt_ledger_refuses_calls_instead_of_restarting_from_zero(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("{not json", encoding="utf-8")
    t = SpendTracker(str(path), cap_usd=D("10"))
    with pytest.raises(SpendLedgerError, match="unreadable"):
        t.check(D("0"))
    with pytest.raises(SpendLedgerError):
        t.record(D("0.1"))
    assert path.read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("content", ['{"total_usd": "-1"}', '{"total_usd": "abc"}', "[]", '{"calls": 1}', '{"total_usd": "NaN"}'])
def test_a_ledger_with_a_bad_total_is_refused(tmp_path, content):
    path = tmp_path / "spend.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SpendLedgerError):
        SpendTracker(str(path), cap_usd=D("10")).check(D("0"))


def test_record_rejects_a_negative_cost(tmp_path):
    with pytest.raises(SpendLedgerError, match="negative"):
        make(tmp_path).record(D("-0.01"))


def test_the_ledger_directory_is_created_on_first_write(tmp_path):
    t = SpendTracker(str(tmp_path / "deep" / "dir" / "spend.json"), cap_usd=D("10"))
    t.record(D("0.01"))
    assert os.path.isfile(t.path)


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "NaN", "Infinity", ""])
def test_cap_values_must_be_positive_finite_numbers(bad):
    with pytest.raises(ValueError, match="USD"):
        spend.parse_usd(bad, "AP_LLM_SPEND_CAP_USD")


def test_parse_usd_reads_decimal_text_exactly():
    assert spend.parse_usd("10", "x") == D("10")
    assert spend.parse_usd(" 0.05 ", "x") == D("0.05")
    assert spend.parse_usd(0.1, "x") == D("0.1")


def test_the_default_ledger_is_the_gitignored_eval_out_directory():
    assert spend.DEFAULT_LEDGER_PATH.endswith(os.path.join("eval", "out", "spend.json"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".gitignore"), encoding="utf-8") as f:
        assert "eval/out/" in f.read().split()

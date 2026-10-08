# ABOUTME: Tests for the live-LLM spend counter: pricing arithmetic, reserve-then-settle admission and ledger persistence.
# ABOUTME: Arithmetic, temp-file, thread and subprocess tests; no network and no SDK client is created.
import json
import os
import subprocess
import sys
import threading
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


def test_a_response_with_no_usage_is_refused_not_priced_at_zero():
    with pytest.raises(SpendLedgerError, match="no usage"):
        cost_usd(MODEL, None)


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


# --- the tracker: ledger, reservations, cap and run cap ----------------------------------------------------------------------


def make(tmp_path, cap="10", run_cap=None, name="spend.json"):
    return SpendTracker(str(tmp_path / name), cap_usd=D(cap), run_cap_usd=None if run_cap is None else D(run_cap))


def ledger_file(tracker):
    with open(tracker.path, encoding="utf-8") as f:
        return json.load(f)


def test_a_new_ledger_starts_at_zero_and_creates_no_file_until_something_is_recorded(tmp_path):
    t = make(tmp_path)
    assert t.total_usd == 0 and t.run_spent_usd == 0 and t.reserved_usd == 0
    assert not os.path.exists(t.path)


def test_recording_adds_to_the_total_and_writes_the_ledger(tmp_path):
    t = make(tmp_path)
    assert t.record(D("0.25")) == D("0.25")
    assert t.record(D("0.5")) == D("0.75")
    data = ledger_file(t)
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
    assert ledger_file(second)["calls"] == 3


def test_many_small_records_do_not_drift(tmp_path):
    t = make(tmp_path)
    for _ in range(1000):
        t.record(D("0.000035"))
    assert make(tmp_path).total_usd == D("0.035")


def test_the_cap_error_is_a_provider_error_so_the_pipeline_routes_it_to_human_review():
    assert issubclass(SpendCapExceeded, LLMProviderError)
    assert issubclass(SpendLedgerError, LLMProviderError)


# reserve


def test_a_reservation_is_written_to_the_ledger_before_reserve_returns(tmp_path):
    t = make(tmp_path)
    r = t.reserve(D("0.4"))
    assert r.amount_usd == D("0.4")
    data = ledger_file(t)
    assert D(data["total_usd"]) == D("0.4")
    assert D(data["reservations"][r.id]["usd"]) == D("0.4")
    assert t.reserved_usd == D("0.4") and t.run_spent_usd == D("0.4")


def test_a_call_that_stays_under_the_cap_is_admitted_and_the_cap_itself_is_reachable(tmp_path):
    t = make(tmp_path, cap="10")
    t.record(D("9.5"))
    t.reserve(D("0.5"))  # total + estimate == cap does not exceed it
    assert t.total_usd == D("10.0")


def test_a_call_that_would_pass_the_cap_is_refused_and_nothing_is_written(tmp_path):
    t = make(tmp_path, cap="10")
    t.record(D("9.5"))
    before = ledger_file(t)
    with pytest.raises(SpendCapExceeded, match="spend cap"):
        t.reserve(D("0.5001"))
    assert ledger_file(t) == before
    assert t.run_spent_usd == D("9.5")


def test_once_the_cap_is_reached_even_a_zero_estimate_over_the_cap_is_refused(tmp_path):
    t = make(tmp_path, cap="1")
    t.record(D("1.2"))
    with pytest.raises(SpendCapExceeded):
        t.reserve(D("0"))


def test_the_total_from_earlier_runs_counts_against_the_cap(tmp_path):
    make(tmp_path, cap="10").record(D("9.9"))
    with pytest.raises(SpendCapExceeded):
        make(tmp_path, cap="10").reserve(D("0.2"))


def test_a_pending_reservation_counts_against_the_cap_for_the_next_caller(tmp_path):
    first = make(tmp_path, cap="10")
    first.reserve(D("6"))
    with pytest.raises(SpendCapExceeded, match="spend cap"):
        make(tmp_path, cap="10").reserve(D("4.0001"))  # another process: it sees the pending 6
    make(tmp_path, cap="10").reserve(D("4"))


def test_a_pending_reservation_counts_against_the_cap_for_the_same_tracker(tmp_path):
    t = make(tmp_path, cap="10")
    t.reserve(D("6"))
    with pytest.raises(SpendCapExceeded):
        t.reserve(D("4.0001"))


def test_the_global_cap_applies_to_settled_and_reserved_spend_together(tmp_path):
    t = make(tmp_path, cap="10")
    t.record(D("5"))
    t.reserve(D("4"))
    with pytest.raises(SpendCapExceeded):
        t.reserve(D("1.0001"))
    t.reserve(D("1"))


def test_a_negative_estimate_is_refused(tmp_path):
    with pytest.raises(SpendLedgerError, match="negative"):
        make(tmp_path).reserve(D("-1"))


def test_the_run_cap_limits_this_runs_spend_even_when_the_global_cap_is_far_away(tmp_path):
    make(tmp_path, cap="10").record(D("3"))
    t = make(tmp_path, cap="10", run_cap="0.5")
    r = t.reserve(D("0.5"))
    t.settle(r, D("0.3"))
    t.reserve(D("0.2"))
    with pytest.raises(SpendCapExceeded, match="run cap"):
        t.reserve(D("0.0001"))


def test_the_run_cap_counts_this_runs_pending_reservations(tmp_path):
    t = make(tmp_path, cap="10", run_cap="1")
    t.reserve(D("0.7"))
    with pytest.raises(SpendCapExceeded, match="run cap"):
        t.reserve(D("0.3001"))


def test_the_run_cap_does_not_count_other_runs_pending_reservations(tmp_path):
    make(tmp_path, cap="10").reserve(D("4"))
    t = make(tmp_path, cap="10", run_cap="1")
    t.reserve(D("1"))  # the other run's 4 counts against the global cap only


def test_the_global_cap_still_applies_when_the_run_cap_is_not_reached(tmp_path):
    make(tmp_path, cap="10").record(D("9.9"))
    t = make(tmp_path, cap="10", run_cap="5")
    with pytest.raises(SpendCapExceeded, match="spend cap"):
        t.reserve(D("0.2"))


# settle


def test_settling_replaces_the_reservation_with_the_observed_cost(tmp_path):
    t = make(tmp_path)
    r = t.reserve(D("0.4"))
    assert t.settle(r, D("0.05")) == D("0.05")
    data = ledger_file(t)
    assert D(data["total_usd"]) == D("0.05") and data["reservations"] == {} and data["calls"] == 1
    assert t.reserved_usd == 0
    assert t.run_spent_usd == D("0.05") and t.run_calls == 1


def test_settling_frees_the_unused_part_of_the_reservation(tmp_path):
    t = make(tmp_path, cap="1")
    r = t.reserve(D("0.9"))
    with pytest.raises(SpendCapExceeded):
        t.reserve(D("0.2"))
    t.settle(r, D("0.1"))
    t.reserve(D("0.9"))


def test_the_observed_cost_is_booked_in_full_even_when_it_passes_the_reservation(tmp_path):
    t = make(tmp_path, cap="1")
    r = t.reserve(D("0.1"))
    assert t.settle(r, D("0.3")) == D("0.3")
    assert t.total_usd == D("0.3")


def test_a_reservation_that_is_never_settled_stays_counted_as_spent(tmp_path):
    t = make(tmp_path, cap="10")
    t.reserve(D("4"))  # the process "crashes" here: nothing settles it
    later = make(tmp_path, cap="10")
    assert later.total_usd == D("4") and later.reserved_usd == D("4")
    with pytest.raises(SpendCapExceeded):
        later.reserve(D("6.0001"))


def test_settling_one_reservation_leaves_the_others_in_place(tmp_path):
    t = make(tmp_path)
    a, b = t.reserve(D("1")), t.reserve(D("2"))
    t.settle(a, D("0.5"))
    data = ledger_file(t)
    assert list(data["reservations"]) == [b.id] and D(data["total_usd"]) == D("2.5")


def test_settling_a_reservation_the_ledger_no_longer_lists_books_the_cost_and_says_so(tmp_path):
    t = make(tmp_path)
    r = t.reserve(D("1"))
    data = ledger_file(t)
    data["reservations"] = {}
    data["total_usd"] = "0"
    with open(t.path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with pytest.raises(SpendLedgerError, match="not in spend ledger"):
        t.settle(r, D("0.2"))
    assert t.total_usd == D("0.2")


def test_settle_rejects_a_negative_cost(tmp_path):
    t = make(tmp_path)
    r = t.reserve(D("1"))
    with pytest.raises(SpendLedgerError, match="negative"):
        t.settle(r, D("-0.01"))
    assert t.reserved_usd == D("1")


def test_record_rejects_a_negative_cost(tmp_path):
    with pytest.raises(SpendLedgerError, match="negative"):
        make(tmp_path).record(D("-0.01"))


# the file: corruption, migration, I/O failure


def test_a_corrupt_ledger_refuses_calls_instead_of_restarting_from_zero(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("{not json", encoding="utf-8")
    t = SpendTracker(str(path), cap_usd=D("10"))
    with pytest.raises(SpendLedgerError, match="unreadable"):
        t.reserve(D("0"))
    with pytest.raises(SpendLedgerError):
        t.record(D("0.1"))
    with pytest.raises(SpendLedgerError):
        t.settle(spend.Reservation("x", D("1")), D("0.1"))
    assert path.read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize(
    "content",
    [
        '{"total_usd": "-1"}', '{"total_usd": "abc"}', "[]", '{"calls": 1}', '{"total_usd": "NaN"}',
        '{"version": 3, "total_usd": "1"}', '{"version": "2", "total_usd": "1"}', '{"version": true, "total_usd": "1"}',
        '{"version": 2, "total_usd": "1", "reservations": []}',
        '{"version": 2, "total_usd": "1", "reservations": {"a": {"usd": "2"}}}',
        '{"version": 2, "total_usd": "5", "reservations": {"a": {"usd": "-1"}}}',
        '{"version": 2, "total_usd": "5", "reservations": {"a": {"usd": "abc"}}}',
        '{"version": 2, "total_usd": "5", "reservations": {"a": "1"}}',
    ],
)
def test_a_ledger_that_cannot_be_trusted_is_refused(tmp_path, content):
    path = tmp_path / "spend.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SpendLedgerError):
        SpendTracker(str(path), cap_usd=D("10")).reserve(D("0"))
    assert path.read_text(encoding="utf-8") == content


def test_a_version_1_ledger_keeps_its_total_and_is_rewritten_as_version_2(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text(json.dumps({"version": 1, "total_usd": "3.25", "calls": 7, "updated_at": "2026-10-07T00:00:00Z"}), encoding="utf-8")
    t = SpendTracker(str(path), cap_usd=D("10"))
    assert t.total_usd == D("3.25") and t.reserved_usd == 0
    r = t.reserve(D("1"))
    t.settle(r, D("0.5"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == spend.LEDGER_VERSION == 2
    assert D(data["total_usd"]) == D("3.75") and data["calls"] == 8 and data["reservations"] == {}


def test_a_version_1_total_counts_against_the_cap(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text(json.dumps({"version": 1, "total_usd": "9.9", "calls": 1}), encoding="utf-8")
    with pytest.raises(SpendCapExceeded):
        SpendTracker(str(path), cap_usd=D("10")).reserve(D("0.2"))


def test_the_ledger_directory_is_created_on_first_write(tmp_path):
    t = SpendTracker(str(tmp_path / "deep" / "dir" / "spend.json"), cap_usd=D("10"))
    t.record(D("0.01"))
    assert os.path.isfile(t.path)


def _boom(*args, **kwargs):
    raise OSError(28, "No space left on device")


@pytest.mark.parametrize("action", ["reserve", "settle", "record", "total_usd"])
@pytest.mark.parametrize("failure", ["replace", "mkstemp", "flock"])
def test_an_io_failure_is_a_spend_ledger_error_not_a_raw_oserror(tmp_path, monkeypatch, action, failure):
    t = make(tmp_path)
    r = t.reserve(D("1"))  # a good ledger on disk first
    target = {"replace": (spend.os, "replace"), "mkstemp": (spend.tempfile, "mkstemp"), "flock": (spend.fcntl, "flock")}[failure]
    monkeypatch.setattr(*target, _boom)
    calls = {
        "reserve": lambda: t.reserve(D("1")),
        "settle": lambda: t.settle(r, D("0.1")),
        "record": lambda: t.record(D("0.1")),
        "total_usd": lambda: t.total_usd,
    }
    expected_to_write = action != "total_usd"
    if not expected_to_write and failure != "flock":
        calls[action]()  # reading needs no temp file and no replace
        return
    with pytest.raises(SpendLedgerError) as info:
        calls[action]()
    assert isinstance(info.value, LLMProviderError)


def test_a_ledger_directory_that_cannot_be_created_is_a_spend_ledger_error(tmp_path):
    (tmp_path / "file").write_text("x", encoding="utf-8")
    t = SpendTracker(str(tmp_path / "file" / "spend.json"), cap_usd=D("10"))
    with pytest.raises(SpendLedgerError, match="lock"):
        t.reserve(D("1"))


def test_a_failed_write_leaves_the_ledger_and_the_run_counters_untouched_and_no_temp_file(tmp_path, monkeypatch):
    t = make(tmp_path)
    r = t.reserve(D("1"))
    before = (tmp_path / "spend.json").read_text(encoding="utf-8")
    with monkeypatch.context() as patched:
        patched.setattr(spend.os, "replace", _boom)
        with pytest.raises(SpendLedgerError):
            t.settle(r, D("0.1"))
        with pytest.raises(SpendLedgerError):
            t.reserve(D("1"))
    assert (tmp_path / "spend.json").read_text(encoding="utf-8") == before
    assert sorted(os.listdir(tmp_path)) == ["spend.json", "spend.json.lock"]
    assert t.run_spent_usd == D("1") and t.run_calls == 0  # the failed settle still holds its reservation


# concurrent admission: never more than the cap, however the callers interleave

CAP, ESTIMATE = D("1.00"), D("0.03")
FITS = 33  # floor(1.00 / 0.03)


def _hammer(trackers, attempts):
    """Each tracker's thread tries `attempts` reservations of ESTIMATE, all released together by a barrier."""
    barrier = threading.Barrier(len(trackers))
    admitted = []
    errors = []

    def work(tracker):
        barrier.wait()
        for _ in range(attempts):
            try:
                tracker.reserve(ESTIMATE)
                admitted.append(1)
            except SpendCapExceeded:
                pass
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in trackers]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    return len(admitted)


def test_threads_with_their_own_trackers_never_admit_more_than_the_cap_between_them(tmp_path):
    trackers = [make(tmp_path, cap=str(CAP)) for _ in range(12)]
    assert _hammer(trackers, attempts=10) == FITS
    check = make(tmp_path, cap=str(CAP))
    assert check.total_usd == ESTIMATE * FITS <= CAP
    assert len(ledger_file(check)["reservations"]) == FITS


def test_threads_sharing_one_tracker_never_pass_the_global_cap_or_the_run_cap(tmp_path):
    shared = make(tmp_path, cap="50", run_cap=str(CAP))
    assert _hammer([shared] * 12, attempts=10) == FITS
    assert shared.run_spent_usd == ESTIMATE * FITS <= CAP
    assert shared.total_usd == ESTIMATE * FITS


def test_concurrent_settling_and_reserving_keeps_the_ledger_consistent(tmp_path):
    t = make(tmp_path, cap="1000")
    barrier = threading.Barrier(8)

    def work():
        barrier.wait()
        for _ in range(20):
            r = t.reserve(D("0.5"))
            t.settle(r, D("0.25"))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.total_usd == D("0.25") * 160 and t.reserved_usd == 0
    assert ledger_file(t)["calls"] == 160 and t.run_calls == 160


_CHILD = """
import sys
from decimal import Decimal
sys.path.insert(0, sys.argv[1])
from ap_invoice_processor.llm.spend import SpendCapExceeded, SpendTracker
tracker = SpendTracker(sys.argv[2], cap_usd=Decimal("1.00"))
admitted = 0
for _ in range(20):
    try:
        tracker.reserve(Decimal("0.03"))
        admitted += 1
    except SpendCapExceeded:
        pass
print(admitted)
"""


def test_separate_processes_never_admit_more_than_the_cap_between_them(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger = str(tmp_path / "spend.json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    procs = [
        subprocess.Popen([sys.executable, "-I", "-c", _CHILD, root, ledger], stdout=subprocess.PIPE, text=True, env=env)
        for _ in range(6)
    ]
    outputs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0] * 6
    assert sum(int(o) for o in outputs) == FITS
    assert SpendTracker(ledger, cap_usd=CAP).total_usd == ESTIMATE * FITS <= CAP


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

# ABOUTME: Spend counter for live LLM calls: prices each response from its usage and keeps a persistent running total.
# ABOUTME: Refuses a call before it is sent when the total plus a conservative estimate would pass the cap or the run cap.
import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterator, Optional

from ap_invoice_processor.llm.provider import LLMProviderError

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# eval/out/ is gitignored, so the ledger (a running record of real spend) is never committed.
DEFAULT_LEDGER_PATH = os.path.join(_ROOT, "eval", "out", "spend.json")
_MTOK = Decimal(1_000_000)
LEDGER_VERSION = 1
# The pre-call estimate assumes one input token per this many UTF-8 bytes. Real text runs near 3 to 4 bytes per
# token (Latin) and 1.5 to 3 (CJK), so 2 over-counts and the check errs toward refusing.
ESTIMATE_BYTES_PER_TOKEN = 2


class SpendError(LLMProviderError):
    """A spend-counter refusal. It is a provider error, so the pipeline treats it like any failed model call."""


class SpendCapExceeded(SpendError):
    """The call was not sent because it could push spending past the cap or the run cap."""


class SpendLedgerError(SpendError):
    """The ledger or a usage figure cannot be trusted (unreadable file, unpriced model, malformed count)."""


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal


@dataclass(frozen=True)
class ModelPricing:
    standard: Rates
    long_context: Rates
    # A prompt of more than this many tokens (fresh + cache read + cache written) is billed at the long-context rates.
    long_context_above_tokens: int


# Source: https://platform.claude.com/docs/en/about-claude/pricing, "Model pricing" table, Claude Haiku 5.5 rows
# ("for prompts up to 100,000 tokens" / "for prompts over 100,000 tokens"), fetched 2026-10-07 and matching
# https://platform.claude.com/docs/en/models/haiku-5-5/overview. Cache read is 0.1x base input and a 5m cache write
# 1.25x; the 1h cache write is 2x. Batch pricing (50% off) is not used here. A model missing from this table cannot
# be called through the counter.
PRICING: Dict[str, ModelPricing] = {
    "claude-haiku-5-5": ModelPricing(
        standard=Rates(Decimal("0.10"), Decimal("0.50"), Decimal("0.01"), Decimal("0.125"), Decimal("0.20")),
        long_context=Rates(Decimal("0.50"), Decimal("2.50"), Decimal("0.05"), Decimal("0.625"), Decimal("1.00")),
        long_context_above_tokens=100_000,
    ),
}


def parse_usd(value: Any, label: str) -> Decimal:
    """A positive, finite dollar amount from text or a number. Raises ValueError naming `label` otherwise."""
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount <= 0:
        raise ValueError(f"{label} must be a positive number of USD, got {value!r}")
    return amount


def _pricing(model: str) -> ModelPricing:
    try:
        return PRICING[model]
    except KeyError:
        raise SpendLedgerError(f"no price known for model {model!r}; refusing to call a model the counter cannot price") from None


def _field(usage: Any, name: str) -> Any:
    return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)


def _tokens(usage: Any, name: str) -> int:
    value = _field(usage, name)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpendLedgerError(f"usage field {name} is not a valid token count: {value!r}")
    return value


def cost_usd(model: str, usage: Any) -> Decimal:
    """Price one response from its `usage` (an SDK Usage object or a mapping with the same field names).

    The prompt length that picks the rate tier is fresh input + cache reads + cache writes; every component, output
    included, is then billed at that tier's rates.
    """
    pricing = _pricing(model)
    fresh = _tokens(usage, "input_tokens")
    out = _tokens(usage, "output_tokens")
    cache_read = _tokens(usage, "cache_read_input_tokens")
    cache_write = _tokens(usage, "cache_creation_input_tokens")

    write_5m = write_1h = 0
    breakdown = _field(usage, "cache_creation")
    if breakdown is not None:
        write_5m = _tokens(breakdown, "ephemeral_5m_input_tokens")
        write_1h = _tokens(breakdown, "ephemeral_1h_input_tokens")
    else:
        write_5m = cache_write
    # Written tokens the TTL breakdown does not account for are priced at the dearer 1h rate.
    write_1h += max(0, cache_write - write_5m - write_1h)

    prompt_tokens = fresh + cache_read + cache_write
    rates = pricing.long_context if prompt_tokens > pricing.long_context_above_tokens else pricing.standard
    total = (
        Decimal(fresh) * rates.input
        + Decimal(out) * rates.output
        + Decimal(cache_read) * rates.cache_read
        + Decimal(write_5m) * rates.cache_write_5m
        + Decimal(write_1h) * rates.cache_write_1h
    )
    return total / _MTOK


def estimate_cost_usd(model: str, prompt_text: str, max_output_tokens: int, extra_input_chars: int = 0) -> Decimal:
    """A deliberately high estimate of one call's cost, computed before it is sent.

    Input is estimated from UTF-8 bytes (see ESTIMATE_BYTES_PER_TOKEN) over the prompt plus `extra_input_chars` of
    anything else the request carries (system prompt, output schema); output is priced at its `max_output_tokens`
    ceiling, so a response can only cost less than this. Prompt caching is ignored (fresh-input rates are the
    highest input rates).
    """
    pricing = _pricing(model)
    nbytes = len(prompt_text.encode("utf-8")) + max(0, extra_input_chars)
    input_tokens = math.ceil(nbytes / ESTIMATE_BYTES_PER_TOKEN)
    rates = pricing.long_context if input_tokens > pricing.long_context_above_tokens else pricing.standard
    return (Decimal(input_tokens) * rates.input + Decimal(max_output_tokens) * rates.output) / _MTOK


class SpendTracker:
    """A persistent running total of spend, with a cap on that total and an optional cap on this run's spend.

    The total lives in a JSON ledger so it survives between runs; each instance also counts what it recorded itself
    (`run_spent_usd`). `check` is the pre-call gate and `record` books the real cost afterwards. Reads and writes
    take an exclusive file lock and replace the ledger atomically. The gap between `check` and `record` is not
    locked, so concurrent processes can overshoot by up to one call each; Anthropic's own workspace cap is the
    backstop for that and for any call whose cost is never observed (a timeout after the request was accepted).
    """

    def __init__(self, path: str, cap_usd: Decimal, run_cap_usd: Optional[Decimal] = None):
        self.path = path
        self.cap_usd = cap_usd
        self.run_cap_usd = run_cap_usd
        self.run_spent_usd = Decimal(0)
        self.run_calls = 0

    # -- ledger file ------------------------------------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> Dict[str, Any]:
        """The ledger contents; a missing file is an empty ledger and anything unreadable is an error."""
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"total_usd": "0", "calls": 0}
        except (OSError, ValueError) as exc:
            raise SpendLedgerError(f"spend ledger {self.path} is unreadable ({exc}); not resetting it to zero") from exc
        try:
            total = Decimal(data["total_usd"])
            calls = data.get("calls", 0)
            if not total.is_finite() or total < 0 or isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                raise ValueError("out-of-range value")
        except (TypeError, KeyError, ValueError, InvalidOperation, AttributeError) as exc:
            raise SpendLedgerError(f"spend ledger {self.path} holds no valid total_usd/calls; not resetting it") from exc
        return {"total_usd": total, "calls": calls}

    def _write(self, total: Decimal, calls: int) -> None:
        payload = {
            "version": LEDGER_VERSION,
            "total_usd": format(total, "f"),
            "calls": calls,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".spend-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # -- public -----------------------------------------------------------------------------------------------------

    @property
    def total_usd(self) -> Decimal:
        with self._locked():
            return Decimal(self._read()["total_usd"])

    def check(self, estimate_usd: Decimal) -> None:
        """Raise SpendCapExceeded unless a call costing up to `estimate_usd` still fits under both caps."""
        total = self.total_usd
        if total + estimate_usd > self.cap_usd:
            raise SpendCapExceeded(
                f"spend cap ${self.cap_usd} would be exceeded: ${total} spent so far + up to ${estimate_usd:.6f} for this call"
            )
        if self.run_cap_usd is not None and self.run_spent_usd + estimate_usd > self.run_cap_usd:
            raise SpendCapExceeded(
                f"run cap ${self.run_cap_usd} would be exceeded: ${self.run_spent_usd} spent this run + "
                f"up to ${estimate_usd:.6f} for this call"
            )

    def record(self, cost_usd_: Decimal) -> Decimal:
        """Add a real, observed cost to the ledger and return the new total."""
        if cost_usd_ < 0:
            raise SpendLedgerError(f"refusing to record a negative cost: {cost_usd_}")
        with self._locked():
            current = self._read()
            total = Decimal(current["total_usd"]) + cost_usd_
            self._write(total, current["calls"] + 1)
        self.run_spent_usd += cost_usd_
        self.run_calls += 1
        return total

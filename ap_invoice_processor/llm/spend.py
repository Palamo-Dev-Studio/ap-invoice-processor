# ABOUTME: Spend counter for live LLM calls: prices each response from its usage and keeps a persistent running total.
# ABOUTME: Admits a call only by reserving its conservative estimate under the ledger lock; settling replaces it with the real cost.
import fcntl
import json
import math
import os
import tempfile
import threading
import uuid
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
# Version 2 adds the `reservations` map. A version 1 file (settled spend only) reads as a version 2 file with none.
LEDGER_VERSION = 2
SUPPORTED_LEDGER_VERSIONS = frozenset({1, 2})
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
    if usage is None:
        raise SpendLedgerError("the response carried no usage figures, so its cost is unknown")
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


@dataclass(frozen=True)
class Reservation:
    """A sum set aside in the ledger for one call before it is sent. `settle` replaces it with the observed cost."""

    id: str
    amount_usd: Decimal


class SpendTracker:
    """A persistent running total of spend, with a cap on that total and an optional cap on this run's spend.

    Admission is reserve then settle. `reserve(estimate)` runs under the ledger's exclusive file lock: it admits the
    call only if the ledger's counted spend (settled costs plus every reservation not yet settled, by any process)
    plus the estimate stays within the cap, and this run's counted spend plus the estimate stays within the run cap;
    it then writes the reservation into the ledger before the caller sends anything. `settle` replaces a reservation
    with the observed cost. A reservation that is never settled (a timeout, a dropped connection, a crash) stays in
    the ledger as spent, because the request may have been billed whether or not its cost was seen. Nothing here
    releases a reservation.

    The ledger is a JSON file that survives between runs. `total_usd` in it is the counted spend: settled costs plus
    the reservations listed under `reservations`. A version 1 file (no reservations) is read as settled spend only.
    Each instance also counts its own spend (`run_spent_usd`: what it settled plus what it still holds reserved).
    Reads and writes replace the ledger atomically, and every I/O failure surfaces as SpendLedgerError.
    """

    def __init__(self, path: str, cap_usd: Decimal, run_cap_usd: Optional[Decimal] = None):
        self.path = path
        self.cap_usd = cap_usd
        self.run_cap_usd = run_cap_usd
        self.run_calls = 0
        self._memory = threading.Lock()
        self._run_settled_usd = Decimal(0)
        self._run_held: Dict[str, Decimal] = {}

    # -- ledger file ------------------------------------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the ledger's exclusive lock; a failure to take it is a SpendLedgerError, not a raw OSError."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            lock = open(self.path + ".lock", "a")
        except OSError as exc:
            raise SpendLedgerError(f"cannot open the lock for spend ledger {self.path}: {exc}") from exc
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            except OSError as exc:
                raise SpendLedgerError(f"cannot lock spend ledger {self.path}: {exc}") from exc
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
            return {"total_usd": Decimal(0), "calls": 0, "reservations": {}}
        except (OSError, ValueError) as exc:
            raise SpendLedgerError(f"spend ledger {self.path} is unreadable ({exc}); not resetting it to zero") from exc
        try:
            version = data.get("version", 1)
            if isinstance(version, bool) or version not in SUPPORTED_LEDGER_VERSIONS:
                raise SpendLedgerError(
                    f"spend ledger {self.path} is version {version!r}; this code reads versions "
                    f"{sorted(SUPPORTED_LEDGER_VERSIONS)} only, and will not guess at the rest"
                )
            total = Decimal(data["total_usd"])
            calls = data.get("calls", 0)
            if not total.is_finite() or total < 0 or isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                raise ValueError("out-of-range value")
            reservations: Dict[str, Dict[str, Any]] = {}
            raw_reservations = data.get("reservations", {})
            if not isinstance(raw_reservations, dict):
                raise ValueError("reservations is not a mapping")
            for key, entry in raw_reservations.items():
                amount = Decimal(entry["usd"])
                if not amount.is_finite() or amount < 0:
                    raise ValueError("out-of-range reservation")
                reservations[str(key)] = {"usd": amount, "at": str(entry.get("at", ""))}
            if sum((r["usd"] for r in reservations.values()), Decimal(0)) > total:
                raise ValueError("reservations exceed the total")
        except SpendLedgerError:
            raise
        except (TypeError, KeyError, ValueError, InvalidOperation, AttributeError) as exc:
            raise SpendLedgerError(f"spend ledger {self.path} holds no valid total_usd/calls/reservations; not resetting it") from exc
        return {"total_usd": total, "calls": calls, "reservations": reservations}

    def _write(self, ledger: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "version": LEDGER_VERSION,
            "total_usd": format(ledger["total_usd"], "f"),
            "calls": ledger["calls"],
            "reservations": {key: {"usd": format(r["usd"], "f"), "at": r["at"]} for key, r in ledger["reservations"].items()},
            "updated_at": now,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".spend-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
                f.write("\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            raise SpendLedgerError(f"could not write spend ledger {self.path}: {exc}") from exc
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    # -- public -----------------------------------------------------------------------------------------------------

    @property
    def total_usd(self) -> Decimal:
        """Counted spend: settled costs plus reservations not yet settled."""
        with self._locked():
            return self._read()["total_usd"]

    @property
    def reserved_usd(self) -> Decimal:
        """The part of `total_usd` still held as reservations (in-flight calls and calls whose outcome is unknown)."""
        with self._locked():
            return sum((r["usd"] for r in self._read()["reservations"].values()), Decimal(0))

    @property
    def run_spent_usd(self) -> Decimal:
        """What this instance has settled plus what it still holds reserved."""
        with self._memory:
            return self._run_settled_usd + sum(self._run_held.values(), Decimal(0))

    def reserve(self, estimate_usd: Decimal) -> Reservation:
        """Admit one call costing up to `estimate_usd`, or raise SpendCapExceeded without writing anything.

        The check and the write happen under one hold of the ledger lock, so concurrent callers (threads or
        processes) can never admit more than the cap between them.
        """
        if estimate_usd < 0:
            raise SpendLedgerError(f"refusing to reserve a negative estimate: {estimate_usd}")
        with self._locked():
            ledger = self._read()
            total = ledger["total_usd"]
            if total + estimate_usd > self.cap_usd:
                raise SpendCapExceeded(
                    f"spend cap ${self.cap_usd} would be exceeded: ${total} spent or reserved so far + "
                    f"up to ${estimate_usd:.6f} for this call"
                )
            run_spent = self.run_spent_usd
            if self.run_cap_usd is not None and run_spent + estimate_usd > self.run_cap_usd:
                raise SpendCapExceeded(
                    f"run cap ${self.run_cap_usd} would be exceeded: ${run_spent} spent or reserved this run + "
                    f"up to ${estimate_usd:.6f} for this call"
                )
            reservation = Reservation(uuid.uuid4().hex, estimate_usd)
            ledger["total_usd"] = total + estimate_usd
            ledger["reservations"][reservation.id] = {
                "usd": estimate_usd,
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self._write(ledger)
            with self._memory:
                self._run_held[reservation.id] = estimate_usd
        return reservation

    def settle(self, reservation: Reservation, actual_usd: Decimal) -> Decimal:
        """Replace `reservation` with the observed cost `actual_usd` and return the ledger's new counted total.

        The observed cost is written in full even when it exceeds the reservation. If the ledger no longer lists the
        reservation (it was edited or restored from a backup) the cost is still booked, and SpendLedgerError says so.
        """
        if actual_usd < 0:
            raise SpendLedgerError(f"refusing to record a negative cost: {actual_usd}")
        with self._locked():
            ledger = self._read()
            held = ledger["reservations"].pop(reservation.id, None)
            ledger["total_usd"] += actual_usd - (held["usd"] if held else Decimal(0))
            ledger["calls"] += 1
            self._write(ledger)
            with self._memory:
                self._run_held.pop(reservation.id, None)
                self._run_settled_usd += actual_usd
                self.run_calls += 1
        if held is None:
            raise SpendLedgerError(
                f"reservation {reservation.id} is not in spend ledger {self.path}; the observed cost ${actual_usd} was booked anyway"
            )
        return ledger["total_usd"]

    def record(self, cost_usd_: Decimal) -> Decimal:
        """Book an observed cost that had no reservation and return the new counted total. It admits nothing."""
        if cost_usd_ < 0:
            raise SpendLedgerError(f"refusing to record a negative cost: {cost_usd_}")
        with self._locked():
            ledger = self._read()
            ledger["total_usd"] += cost_usd_
            ledger["calls"] += 1
            self._write(ledger)
            with self._memory:
                self._run_settled_usd += cost_usd_
                self.run_calls += 1
        return ledger["total_usd"]

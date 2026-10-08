# ABOUTME: Shared pytest setup: registers the `live` marker and keeps every non-live test away from credentials, the real ledger and the network.
# ABOUTME: Only tests marked `live` (opt-in, AP_LIVE=1) may see the machine's key file and provider environment or open a socket.
import socket

import pytest

# Variables that choose or configure a provider, or that hold credentials. A developer machine may set any of them.
_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "AP_LLM_PROVIDER",
    "AP_LLM_MODEL",
    "AP_LLM_SPEND_CAP_USD",
    "AP_LLM_EFFORT",
    "AP_LLM_MAX_TOKENS",
    "AP_OCR_LANGS",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: calls the real Anthropic API and spends money; runs only when AP_LIVE=1 is set"
    )


@pytest.fixture(autouse=True)
def isolate_provider_credentials(request, monkeypatch, tmp_path_factory, tmp_path):
    """Hide the real key, key file and spend ledger from every test except those marked `live`."""
    if request.node.get_closest_marker("live"):
        return
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AP_INTAKE_ENV_FILE", str(tmp_path_factory.getbasetemp() / "no-such-credentials.env"))
    # A provider built with the default ledger path would otherwise reserve spend in the developer's real ledger.
    monkeypatch.setattr("ap_invoice_processor.llm.anthropic_provider.DEFAULT_LEDGER_PATH", str(tmp_path / "default-spend.json"))


@pytest.fixture(autouse=True)
def blocked_network_attempts(request, monkeypatch):
    """Refuse every outbound connection and DNS lookup in a non-live test, and fail the test if one was attempted.

    The refusal alone is not enough: code that catches connection errors (the live provider turns them into a
    fallback) would let a test pass while having tried to reach a server. So each attempt is recorded and the test
    errors at teardown unless it acknowledged the attempts by clearing the list this fixture returns. Local
    socketpairs, which event loops use, are untouched.
    """
    attempts = []
    if request.node.get_closest_marker("live"):
        yield attempts
        return

    def refuse(what):
        def blocked(*args, **kwargs):
            attempts.append(what)
            raise OSError(f"network access is blocked in tests that are not marked live: {what}")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", refuse("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("socket.getaddrinfo"))
    yield attempts
    if attempts:
        pytest.fail(
            f"network access was attempted {len(attempts)} time(s) by a test not marked live ({', '.join(sorted(set(attempts)))}); "
            "mock the transport, or mark the test live",
            pytrace=False,
        )

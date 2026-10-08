# ABOUTME: Tests for the suite's own isolation (tests/conftest.py): no credentials, no real ledger, no network for non-live tests.
# ABOUTME: Several checks run a throwaway pytest in a temp directory with a polluted environment, so they prove the fixture itself.
import os
import shutil
import socket
import subprocess
import sys
import textwrap

import anthropic
import pytest

from ap_invoice_processor.llm import anthropic_provider as ap
from ap_invoice_processor.llm.provider import LLMProviderError
from ap_invoice_processor.llm.spend import DEFAULT_LEDGER_PATH, SpendTracker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SENTINEL_KEY = "sk-ant-ISOLATION-SENTINEL-not-a-real-key"


# --- the network block, exercised in this process -----------------------------------------------------------------------


def test_opening_a_connection_is_refused_and_recorded(blocked_network_attempts):
    with pytest.raises(OSError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with pytest.raises(OSError, match="network access is blocked"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))
    with pytest.raises(OSError, match="network access is blocked"):
        socket.getaddrinfo("localhost", 9)
    assert len(blocked_network_attempts) >= 3
    blocked_network_attempts.clear()  # acknowledged: these were the point of the test


def test_a_real_sdk_client_that_is_not_mocked_cannot_reach_a_server(tmp_path, blocked_network_attempts):
    # base_url is a closed loopback port, so even if the block were missing this test could not reach a real API.
    client = anthropic.Anthropic(api_key="not-a-real-key", base_url="http://127.0.0.1:9", max_retries=0)
    provider = ap.AnthropicProvider(client, SpendTracker(str(tmp_path / "spend.json"), cap_usd=ap.parse_usd("10", "cap")))
    with pytest.raises(LLMProviderError, match="connect"):
        provider.complete("extract", "p", "doc-1")
    assert blocked_network_attempts, "the provider call never reached the socket layer, so nothing proves it is blocked"
    blocked_network_attempts.clear()


def test_a_unix_socket_pair_used_by_event_loops_is_not_blocked():
    a, b = socket.socketpair()
    a.close()
    b.close()


# --- credentials and the real ledger ------------------------------------------------------------------------------------


def test_a_non_live_test_cannot_see_credentials_or_the_providers_environment():
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "AP_LLM_PROVIDER", "AP_LLM_SPEND_CAP_USD", "AP_LLM_MODEL"):
        assert name not in os.environ
    assert not os.path.exists(os.environ["AP_INTAKE_ENV_FILE"])


def test_a_provider_built_with_the_default_ledger_never_touches_the_real_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AP_LLM_SPEND_CAP_USD", "10")
    provider = ap.build_anthropic_provider()
    assert provider.tracker.path != DEFAULT_LEDGER_PATH
    assert not os.path.realpath(provider.tracker.path).startswith(os.path.realpath(ROOT) + os.sep)


# --- the fixtures themselves, proved with a throwaway pytest run --------------------------------------------------------


def _run_pytest(tmp_path, test_source, extra_env=None):
    shutil.copy(os.path.join(HERE, "conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(textwrap.dedent(test_source), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"AP_LIVE", "ANTHROPIC_API_KEY", "AP_LLM_PROVIDER", "AP_LLM_SPEND_CAP_USD"}}
    env["PYTHONPATH"] = ROOT
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )


def test_a_swallowed_connection_attempt_still_fails_the_test_loudly(tmp_path):
    result = _run_pytest(
        tmp_path,
        """
        import socket

        def test_swallows_the_error():
            try:
                socket.create_connection(("127.0.0.1", 9), timeout=1)
            except OSError:
                pass  # code that catches a connection error would pass without the teardown check
        """,
    )
    assert result.returncode != 0, result.stdout
    assert "1 passed, 1 error" in result.stdout and "network access" in result.stdout


def test_a_live_marked_test_may_use_the_network(tmp_path):
    result = _run_pytest(
        tmp_path,
        """
        import socket
        import pytest

        @pytest.mark.live
        def test_connects():
            with pytest.raises(ConnectionRefusedError):  # port 9 is closed: refused by the OS, not blocked by us
                socket.create_connection(("127.0.0.1", 9), timeout=1)
        """,
    )
    assert result.returncode == 0, result.stdout


def test_a_machine_with_a_key_a_key_file_and_a_provider_choice_stays_invisible_to_non_live_tests(tmp_path):
    key_file = tmp_path / "real.env"
    key_file.write_text(f"ANTHROPIC_API_KEY={SENTINEL_KEY}\nAP_LLM_SPEND_CAP_USD=5\n", encoding="utf-8")
    os.chmod(key_file, 0o600)
    polluted = {
        "ANTHROPIC_API_KEY": SENTINEL_KEY,
        "ANTHROPIC_AUTH_TOKEN": SENTINEL_KEY,
        "ANTHROPIC_BASE_URL": "https://example.invalid",
        "AP_LLM_PROVIDER": "anthropic",
        "AP_LLM_SPEND_CAP_USD": "5",
        "AP_INTAKE_ENV_FILE": str(key_file),
    }
    probe = """
        import os
        import pytest
        from ap_invoice_processor.llm.provider import get_provider

        def test_non_live_sees_nothing():
            assert not [k for k in os.environ if k.startswith("ANTHROPIC_")]
            assert "AP_LLM_PROVIDER" not in os.environ and "AP_LLM_SPEND_CAP_USD" not in os.environ
            assert not os.path.exists(os.environ["AP_INTAKE_ENV_FILE"])
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                get_provider("anthropic")

        @pytest.mark.live
        def test_live_sees_the_machine():
            assert os.environ["ANTHROPIC_API_KEY"] == "%s"
        """ % SENTINEL_KEY
    result = _run_pytest(tmp_path, probe, polluted)
    assert result.returncode == 0, result.stdout
    assert "2 passed" in result.stdout

import builtins
import os

from swebench_runner import tui


def test_preflight_success(monkeypatch):
    # Simulate successful preflight path
    monkeypatch.setattr(tui, "Confirm", type("C", (), {"ask": staticmethod(lambda *a, **k: True)}))
    monkeypatch.setattr(tui, "_run_harness_preflight", staticmethod(lambda namespace=None, timeout_s=None: (True, "ok")))

    # Should just print success and return
    tui.preflight_wizard()


def test_preflight_failure_and_retry_success(monkeypatch):
    seq = [(False, "denied"), (True, "ok2")]

    def fake_preflight(namespace=None, timeout_s=None):
        return seq.pop(0)

    # First ask True to run preflight, then ask for trying fixes, default True, then provide ns
    monkeypatch.setattr(tui, "Confirm", type("C", (), {"ask": staticmethod(lambda *a, **k: True)}))

    # Simulate user entering namespace then accepting
    answers = iter(["ghcr.io/epoch-research"])  # namespace input
    monkeypatch.setattr(tui, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, **k: next(answers))}))

    monkeypatch.setattr(tui, "_run_harness_preflight", staticmethod(fake_preflight))

    # Avoid writing to .env in tests by redirecting writes to devnull
    orig_open = builtins.open

    def fake_open(file, mode='r', *args, **kwargs):
        if isinstance(file, str) and file.endswith(".env") and 'a' in mode:
            return orig_open(os.devnull, 'a')
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    tui.preflight_wizard()

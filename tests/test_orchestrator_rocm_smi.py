"""Live ROCm telemetry must be parsed fail-closed from the pinned SMI tool."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from automil.backends import _orchestrator_daemon as daemon_module


def test_query_rocm_gpus_parses_total_and_used_vram(monkeypatch):
    payload = {
        "card0": {
            "VRAM Total Memory (B)": "68719476736",
            "VRAM Total Used Memory (B)": "1073741824",
        },
        "card1": {
            "VRAM Total Memory (B)": 68719476736,
            "VRAM Total Used Memory (B)": 2147483648,
        },
    }

    def fake_run(command, **kwargs):
        assert command == [
            daemon_module.ROCM_SMI_PATH,
            "--showmeminfo",
            "vram",
            "--json",
        ]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    devices = daemon_module.query_rocm_gpus()

    assert [device.index for device in devices] == [0, 1]
    assert [device.total_mb for device in devices] == [65536, 65536]
    assert [device.free_mb for device in devices] == [64512, 63488]


def test_query_rocm_gpus_rejects_partial_or_invalid_telemetry(monkeypatch):
    payload = {
        "card0": {
            "VRAM Total Memory (B)": "68719476736",
            "VRAM Total Used Memory (B)": "not-a-number",
        }
    }
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="",
        ),
    )

    assert daemon_module.query_rocm_gpus() == []


def test_query_rocm_gpus_fails_closed_when_probe_times_out(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("rocm-smi", 10)

    monkeypatch.setattr(daemon_module.subprocess, "run", timeout)

    assert daemon_module.query_rocm_gpus() == []

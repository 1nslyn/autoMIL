"""CFG-1 / CFG-2: every key the scaffold writes must be a key something reads.

A config block that nothing consumes is worse than a missing one: it reads as a
knob, an operator sets it, and the setting silently does nothing. Two of these
were found in the 2026-07-28 pass.

**CFG-2** — the template declared ``default_vram_estimate_gb`` and
``max_concurrent_per_gpu`` **twice**: under ``orchestrator:`` as literals, and
again under ``cap:`` as the Jinja expressions ``automil init``'s hardware
healthcheck fills in. The daemon reads only ``orchestrator:``
(``_orchestrator_daemon.py:439-444``), so the measured values went into a dead
location and the conservative literals were what actually ran. This directly
aggravated H-2: ``max_concurrent_per_gpu`` is the knob that decides how much of a
cell's wall-clock budget is spent in parallel, so an arm-correlated value is an
arm-correlated bias on "equal effort".

**CFG-1** — the ``gate:`` block (``auto_nominate`` / ``K`` / ``p_threshold`` /
``bootstrap_reps``) was read by no code at all; every value came from
``automil gate register-manifest``'s CLI defaults. For a **pre-registered**
statistical gate that is the wrong direction of travel: the thresholds should
come from a committed file, not from what an operator typed at the moment they
decided to run the gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TEMPLATE = _REPO / "src" / "automil" / "templates" / "config.yaml.j2"
_DAEMON = _REPO / "src" / "automil" / "backends" / "_orchestrator_daemon.py"

#: Keys the daemon reads off ``orchestrator:``.
_ORCHESTRATOR_KEYS = (
    "poll_interval_sec", "safety_margin_gb", "default_timeout_min",
    "max_concurrent_per_gpu", "default_vram_estimate_gb", "scheduling_policy",
)


def _template() -> str:
    return _TEMPLATE.read_text()


def _block(text: str, name: str) -> str:
    """The lines of one top-level YAML block in the template."""
    out, inside = [], False
    for line in text.splitlines():
        if re.match(rf"^{re.escape(name)}:\s*$", line):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace():
                break
            out.append(line)
    return "\n".join(out)


class TestNoKeyIsDeclaredInTwoPlaces:
    @pytest.mark.parametrize("key", ["default_vram_estimate_gb", "max_concurrent_per_gpu"])
    def test_orchestrator_keys_appear_once(self, key):
        """CFG-2: the cap: copies were dead — the daemon reads orchestrator:."""
        text = _template()
        occurrences = [
            ln for ln in text.splitlines()
            if re.match(rf"^\s+{re.escape(key)}:", ln)
        ]
        assert len(occurrences) == 1, (
            f"{key} is declared {len(occurrences)}x in the scaffold; only the "
            f"orchestrator: copy is read, so any other is a knob that does "
            f"nothing:\n" + "\n".join(occurrences)
        )

    @pytest.mark.parametrize("key", ["default_vram_estimate_gb", "max_concurrent_per_gpu"])
    def test_they_live_under_orchestrator(self, key):
        assert re.search(rf"^\s+{re.escape(key)}:", _block(_template(), "orchestrator"), re.M)

    @pytest.mark.parametrize("key", ["default_vram_estimate_gb", "max_concurrent_per_gpu"])
    def test_they_do_not_live_under_cap(self, key):
        assert not re.search(rf"^\s+{re.escape(key)}:", _block(_template(), "cap"), re.M)

    def test_the_healthcheck_values_reach_the_block_the_daemon_reads(self):
        """The point of the Jinja expressions: `automil init` measures the host.
        Before CFG-2 they were rendered into the dead cap: copy."""
        orch = _block(_template(), "orchestrator")
        assert "default_vram_estimate_gb" in orch and "{{" in orch

    def test_every_orchestrator_key_the_daemon_reads_is_in_the_scaffold(self):
        """Guards the other direction: a key the daemon reads but the scaffold
        never writes is undiscoverable."""
        orch = _block(_template(), "orchestrator")
        missing = [k for k in _ORCHESTRATOR_KEYS if not re.search(rf"^\s+{k}:", orch, re.M)]
        assert not missing, f"daemon reads {missing} but the scaffold does not write them"

    def test_the_daemon_still_reads_them_from_orchestrator(self):
        """Pins the direction of the fix: if the daemon ever moves to cap:, this
        test fails rather than the template silently going stale again."""
        src = _DAEMON.read_text()
        assert 'orch_cfg = self.config.get("orchestrator", {})' in src
        for key in ("max_concurrent_per_gpu", "default_vram_estimate_gb"):
            assert f'orch_cfg.get("{key}"' in src


class TestGateBlockIsActuallyRead:
    """CFG-1: it was decorative — every value came from CLI defaults."""

    def test_register_manifest_reads_the_config(self):
        from automil.gate import config as gate_config

        cfg = gate_config.load_gate_config({"gate": {
            "K": 5, "p_threshold": 0.01, "bootstrap_reps": 4000, "auto_nominate": True,
        }})
        assert (cfg.K, cfg.p_threshold, cfg.bootstrap_reps) == (5, 0.01, 4000)
        assert cfg.auto_nominate is True

    def test_defaults_match_the_previous_cli_defaults(self):
        """No behaviour change for a project whose config omits the block."""
        from automil.gate import config as gate_config

        cfg = gate_config.load_gate_config({})
        assert (cfg.K, cfg.p_threshold, cfg.bootstrap_reps) == (2, 0.05, 1000)
        assert cfg.auto_nominate is False

    def test_a_missing_block_is_not_an_error(self):
        from automil.gate import config as gate_config

        assert gate_config.load_gate_config(None).K == 2

    @pytest.mark.parametrize("bad", [
        {"gate": {"K": 0}},
        {"gate": {"K": -1}},
        {"gate": {"p_threshold": 0}},
        {"gate": {"p_threshold": 1.5}},
        {"gate": {"bootstrap_reps": 50}},   # below manifest.BOOTSTRAP_REPS_FLOOR
        {"gate": {"K": "two"}},
    ])
    def test_invalid_values_fail_loudly(self, bad):
        from automil.gate import config as gate_config

        with pytest.raises(ValueError):
            gate_config.load_gate_config(bad)

    def test_a_non_mapping_gate_block_fails_loudly(self):
        from automil.gate import config as gate_config

        with pytest.raises((ValueError, TypeError)):
            gate_config.load_gate_config({"gate": ["K", 2]})

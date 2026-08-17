"""Campaign attempt-timeout cap: --timeout may lower, never raise (A5).

``PROTOCOL.attempt_timeout`` is a hash-audited failure-containment constant,
yet ``submit --timeout`` wrote ``spec.timeout_min`` unchecked — the
runtime-canary agent raised 360→600 unchallenged. ``enforce_attempt_timeout_cap``
is a standalone check with two explicit enforcing callers: submit and the
daemon's launch-time revalidation (queue specs are agent-editable JSON, so
submit-only enforcement is bypassable). Both pass the RAW
``orchestrator.default_timeout_min`` config value, so an absent key skips the
check SYMMETRICALLY — never accepted at submit and refused post-billing at
launch against a framework fallback the config never declared.
"""
from __future__ import annotations

import pytest

from automil.admissibility import AdmissibilityError, enforce_attempt_timeout_cap


def test_raising_the_timeout_is_refused() -> None:
    with pytest.raises(AdmissibilityError, match="failure containment"):
        enforce_attempt_timeout_cap(600, 360)


def test_lowering_and_matching_pass_the_cap() -> None:
    enforce_attempt_timeout_cap(360, 360)
    enforce_attempt_timeout_cap(90, 360)


def test_absent_reference_or_spec_skips_symmetrically() -> None:
    # Absent config key (None reference) or no explicit --timeout (None spec):
    # skipped — at BOTH gates, by construction of the shared reference.
    enforce_attempt_timeout_cap(None, 360)
    enforce_attempt_timeout_cap(600, None)
    enforce_attempt_timeout_cap(None, None)


def test_both_enforcing_callers_use_the_raw_config_reference() -> None:
    """submit and the daemon must resolve the SAME reference: the raw config
    value. The daemon's scheduling fallback (DEFAULT_TIMEOUT_MIN) must never
    leak into the cap, or a config lacking the key accepts at submit and
    refuses after billing at launch."""
    import inspect

    from automil.backends import _orchestrator_daemon as daemon_mod

    src = inspect.getsource(daemon_mod.ExperimentOrchestrator._load_config_and_state) \
        if hasattr(daemon_mod.ExperimentOrchestrator, "_load_config_and_state") \
        else inspect.getsource(daemon_mod)
    assert 'orch_cfg.get("default_timeout_min")' in src, (
        "the daemon's timeout_cap must be the RAW config value (no fallback)"
    )
    from automil.cli import submit as submit_mod

    submit_src = inspect.getsource(submit_mod)
    assert "enforce_attempt_timeout_cap(" in submit_src
    daemon_src = inspect.getsource(daemon_mod)
    assert "enforce_attempt_timeout_cap(" in daemon_src
    assert "self.timeout_cap" in daemon_src

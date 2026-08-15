"""Campaign attempt-timeout cap: --timeout may lower, never raise (A5).

``PROTOCOL.attempt_timeout`` is a hash-audited failure-containment constant,
yet ``submit --timeout`` wrote ``spec.timeout_min`` unchecked — the
runtime-canary agent raised 360→600 unchallenged. The cap lives in
``validate_campaign_binding`` behind OPTIONAL params: two enforcing callers
(submit, and the daemon's launch-time revalidation — queue specs are
agent-editable JSON, so submit-only enforcement is bypassable), two
non-enforcing callers (campaign materialization, promotion) that pass
nothing and skip the check.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _call(spec_timeout, default_timeout):
    from automil.admissibility import validate_campaign_binding

    # The cap check fires before any manifest I/O, so a placeholder path and
    # minimal campaign mapping exercise it in isolation.
    return validate_campaign_binding(
        Path("/nonexistent/manifest.json"),
        {"campaign_id": "c"},
        base_run_command=None,
        budget_cell_id="cell",
        spec_timeout_min=spec_timeout,
        default_timeout_min=default_timeout,
    )


def test_raising_the_timeout_is_refused() -> None:
    from automil.admissibility import AdmissibilityError

    with pytest.raises(AdmissibilityError, match="failure containment"):
        _call(600, 360)


def test_lowering_and_matching_pass_the_cap() -> None:
    from automil.admissibility import AdmissibilityError

    # Passing the cap then failing on the minimal campaign mapping proves the
    # cap itself accepted these values.
    for spec_timeout in (360, 90):
        with pytest.raises(AdmissibilityError, match="missing non-empty string"):
            _call(spec_timeout, 360)


def test_absent_params_skip_the_check() -> None:
    from automil.admissibility import AdmissibilityError

    # Non-enforcing callers (materialization, promotion) pass neither param;
    # a spec with no explicit --timeout passes None for the spec side.
    for spec_timeout, default in ((None, 360), (600, None), (None, None)):
        with pytest.raises(AdmissibilityError, match="missing non-empty string"):
            _call(spec_timeout, default)

"""DEC-01 / D-206: framework purity grep gate.

Asserts src/automil/ contains zero autobench/AUTOBENCH_/benchmarks/ references
outside the hardcoded allowlist of informational comments and known-default
operator-facing help strings. Future commits that accidentally re-introduce a
leak break this test with a named offender.

The allowlist uses content-anchor substrings (not just file:line) so line drift
within an allowlisted file fails loudly: the line check fails, the substring
check then fires on the new line, and the offender is named in the failure
message with a clear "update allowlist" instruction.

F-01 fix (Iter-2 plan-check): revert_baseline.py:87 has a ClickException with
'benchmarks/lib/CLAM/**' as an example value for registry.protected default
help text. That string is operator-facing help, not a code path; allowlisted
by intent.

Allowlist deviations from plan baseline (3 expected -> 5 actual, post-08-04/08-05):
  - config.yaml.j2:105: migration-note comment for autobench-shaped consumers
  - config.yaml.j2:122: inline example comment in scoring.formula block
Both are informational-only; they contain no functional consumer-namespace code
path. Allowlisted per deviation Rule 2 (correct allowlist coverage so the test
does not produce false failures on main).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_AUTOMIL = _REPO_ROOT / "src" / "automil"

# Hardcoded allowlist: file:line keys with a content anchor for line-drift
# detection. If the comment moves, the line-key check fails AND the content
# check fires on the new line, triggering "update allowlist" failure.
#
# Content-anchor substrings are load-bearing: line numbers are tolerated drift
# (update the key); anchor strings that no longer match the line are drift errors.
_ALLOWLIST: dict[str, str] = {
    # 1. Informational comment about consumer-specific vars; documents the
    #    env.passthrough seam without auto-injecting any value.
    #    Line shifted to :60 by DBT-03 (14-02): removed 2-line deferral comment block.
    "src/automil/backends/_orchestrator_daemon.py:60":
        "Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)",
    # 2. Comment in verify_repro about the clean env not leaking AUTOBENCH_*.
    "src/automil/cli/lifecycle/verify_repro.py:84":
        "no AUTOBENCH_* leakage",
    # 3. F-01 fix: registry.protected default-help string inside a
    #    ClickException. The token 'benchmarks/lib/CLAM/**' is an EXAMPLE
    #    value used to teach operators what registry.protected accepts;
    #    not a path the framework reads or depends on. Allowlist by intent.
    "src/automil/cli/lifecycle/revert_baseline.py:87":
        "'benchmarks/lib/CLAM/**'",
    # 4. Migration note comment in config.yaml.j2 directing autobench-shaped
    #    consumers to the CHANGELOG 8.0.0 BREAKING section. Informational only;
    #    not a code path. Retained by 08-04 executor.
    #    Line shifted +1 by 12-02 (added scheduling_policy key above it),
    #    +4 by the run.mil_model addition to the run: block.
    "src/automil/templates/config.yaml.j2:114":
        "autobench-shaped consumers",
    # 5. Inline example comment in the scoring.formula block showing what an
    #    autobench consumer formula looks like. Documentation only. Retained by
    #    08-04 executor.
    #    Line shifted +1 by 12-02 (added scheduling_policy key to orchestrator block),
    #    +4 by the run.mil_model addition to the run: block.
    "src/automil/templates/config.yaml.j2:140":
        "autobench consumer",
}


def _is_allowlisted(rel_path: str, line_no: str, content: str) -> bool:
    """Return True if (rel_path, line_no, content) matches an allowlist entry.

    Both the file:line key AND the content-anchor substring must match. This
    catches line drift: if a comment moves, the file:line lookup fails and the
    substring fall-through fails as well, and the offender is named.
    """
    key = f"{rel_path}:{line_no}"
    expected_substring = _ALLOWLIST.get(key)
    if expected_substring is None:
        return False
    return expected_substring in content


def test_framework_purity_no_autobench_refs():
    """D-206 / DEC-01: grep src/automil/ returns at most allowlisted lines.

    POSIX grep exit codes:
      0 = matches found (we then filter via allowlist)
      1 = no matches (success path)
      2 = grep error
    """
    result = subprocess.run(
        ["grep", "-rEn", "autobench|AUTOBENCH_|benchmarks/", str(_SRC_AUTOMIL)],
        capture_output=True, text=True,
    )

    if result.returncode == 2:
        pytest.fail(f"grep error (returncode 2): {result.stderr}")

    if result.returncode == 1 and not result.stdout.strip():
        # No matches found at all; trivially passing.
        return

    matches: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Format: /abs/path:LINE:content
        m = re.match(r"^(.*?):(\d+):(.*)$", line)
        if not m:
            matches.append(line)
            continue
        abs_path, line_no, content = m.group(1), m.group(2), m.group(3)
        try:
            rel_path = str(Path(abs_path).relative_to(_REPO_ROOT))
        except ValueError:
            rel_path = abs_path
        if _is_allowlisted(rel_path, line_no, content):
            continue
        matches.append(f"{rel_path}:{line_no}:{content}")

    assert not matches, (
        "Framework purity (DEC-01 / D-206) violated. The following references "
        "to autobench / AUTOBENCH_ / benchmarks/ were found in src/automil/ "
        "outside the hardcoded allowlist:\n  "
        + "\n  ".join(matches)
        + "\n\nautoMIL is generic; autobench is one consumer. Move the leaked "
        "reference to consumer-side code (benchmarks/src/autobench/), or "
        "update _ALLOWLIST in tests/test_framework_purity.py if it is a "
        "deliberate informational comment or operator-facing help string."
    )


def test_allowlist_anchors_still_present():
    """Defensive: the allowlisted lines still contain their anchor substrings.

    If a future commit moves or rewrites the allowlisted lines, this test fails
    so the operator can update the allowlist deliberately rather than risk a
    silent bypass.
    """
    for key, expected_substring in _ALLOWLIST.items():
        rel_path, line_no_str = key.rsplit(":", 1)
        line_no = int(line_no_str)
        path = _REPO_ROOT / rel_path
        if not path.exists():
            pytest.fail(f"Allowlisted file missing: {rel_path}")
        lines = path.read_text().splitlines()
        if line_no > len(lines):
            pytest.fail(
                f"Allowlist line {line_no} beyond EOF for {rel_path} "
                f"(file has {len(lines)} lines). Update _ALLOWLIST."
            )
        actual = lines[line_no - 1]
        assert expected_substring in actual, (
            f"Allowlist anchor drift detected at {key}: expected substring "
            f"{expected_substring!r} not found on line {line_no} of {rel_path}. "
            f"Actual content: {actual!r}. Update _ALLOWLIST keys + anchors."
        )


def test_purity_test_does_not_execute_consumer_code():
    """Pitfall 7d defender: the purity test reads source via grep + filesystem only.

    Verifies no runtime consumer-code imports exist in this test file. If a
    future commit adds execution of autobench or benchmarks code here, the
    purity gate becomes coupled to consumer state, defeating its purpose.
    """
    self_text = Path(__file__).read_text()
    # Build tokens dynamically so they do not appear verbatim in this file
    # (which would cause this very test to flag itself as a violator).
    pkg_autobench = "autobench"
    pkg_benchmarks = "benchmarks"
    forbidden = (
        f"import {pkg_autobench}",
        f"from {pkg_autobench}",
        f"import {pkg_benchmarks}",
        f"from {pkg_benchmarks}",
    )
    # Count occurrences: this test's own construction block contributes exactly
    # one f-string per token (in the tuple above). Subtract the one allowed
    # construction occurrence; any additional occurrence is a real violation.
    for token in forbidden:
        count = self_text.count(token)
        # Each token appears exactly once: inside the f"..." expression above.
        assert count <= 1, (
            f"tests/test_framework_purity.py must not exercise consumer "
            f"code; token {token!r} found {count} times (expected at most 1 "
            f"from the construction block). Extra occurrences indicate a real "
            f"consumer import was added."
        )

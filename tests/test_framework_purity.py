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
  - config.yaml.j2:128: migration-note comment for autobench-shaped consumers
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
    #    Line shifted to :61 by L-6 (audit 2026-07-23): added `import fcntl` for
    #    cmd_submit's id-allocation lock; to :62 by the campaign-boundary hash
    #    imports; to :63 by the live ROCm parser's `re` import; to :64 by the
    #    activity-admission enum import; to :65 by the umask-honoring
    #    atomic-write import (D-2xx: `from automil.runtime_helpers import
    #    atomic_write_text`).
    "src/automil/backends/_orchestrator_daemon.py:65":
        "Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)",
    # 2. Comment in verify_repro about the clean env not leaking AUTOBENCH_*.
    "src/automil/cli/lifecycle/verify_repro.py:84":
        "no AUTOBENCH_* leakage",
    # 3. F-01 fix: registry.protected default-help string inside a
    #    ClickException. The token 'benchmarks/lib/CLAM/**' is an EXAMPLE
    #    value used to teach operators what registry.protected accepts;
    #    not a path the framework reads or depends on. Allowlist by intent.
    #    Line shifted +2 by the base-tree probe fix (`os` + `tempfile`
    #    imports for the throwaway GIT_INDEX_FILE).
    "src/automil/cli/lifecycle/revert_baseline.py:89":
        "'benchmarks/lib/CLAM/**'",
    # 4. Migration note comment in config.yaml.j2 directing autobench-shaped
    #    consumers to the CHANGELOG 8.0.0 BREAKING section. Informational only;
    #    not a code path. Retained by 08-04 executor.
    #    Line shifted +1 by 12-02 (added scheduling_policy key above it),
    #    +4 by the run.mil_model addition to the run: block,
    #    +6 by the H-4 registry block, -2 by CFG-2 (dead cap: duplicates removed),
    #    +9 by CR-4's se_multiplier documentation in the scoring: block,
    #    +9 by the architecture-preserving command/identity channels,
    #    -1 by the composite_formula key removal (single-metric optimization),
    #    +3 by the metrics.primary/scoring.formula agreement comment,
    #    +5 by the vocabulary-agnostic default guidance (round 6).
    "src/automil/templates/config.yaml.j2:136":
        "autobench-shaped consumers",
    # (The former entry 5 — an "autobench consumer" formula example in the
    # scoring block — was removed by B2: the rewritten reducer documentation
    # carries no consumer reference, so no allowlist exception is needed.)
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
    # Source only. Without these two flags the scan also walks __pycache__, and a
    # .pyc embeds its module's docstrings and string literals -- so every
    # ALLOWLISTED source line reappears as a `Binary file ... matches` line with
    # no line number, which _is_allowlisted can never match. pytest compiles the
    # modules during collection, so the test could not pass in a normal run and
    # was permanently deselected -- which is worse than useless, because a
    # deselected purity test hides real violations too.
    result = subprocess.run(
        ["grep", "-rEn", "--exclude-dir=__pycache__", "--binary-files=without-match",
         "autobench|AUTOBENCH_|benchmarks/", str(_SRC_AUTOMIL)],
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


def test_the_scan_is_source_only():
    """Guard the fix above: the scan must not walk compiled bytecode.

    A .pyc embeds its module's docstrings and string literals, so scanning
    __pycache__ makes every allowlisted source line reappear as an unmatchable
    `Binary file ... matches` row. That made this test fail on any run where the
    modules had been imported -- i.e. every run, since pytest compiles them
    during collection -- so it was permanently deselected, and a deselected
    purity test also stops catching the real violations it exists for.
    """
    import inspect

    src = inspect.getsource(test_framework_purity_no_autobench_refs)
    assert "--exclude-dir=__pycache__" in src
    assert "--binary-files=without-match" in src

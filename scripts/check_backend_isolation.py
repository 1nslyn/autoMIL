#!/usr/bin/env python3
"""BCK-04 / D-64: Backend isolation lint.

Forbids ``os.kill | os.killpg | os.getpid | Popen | .pid`` references outside
the allowlist.  Pre-commit hook + always-on pytest enforcement (D-65).

ALLOWLIST rationale:
  - ``backends/local.py``                — job-control surface (BCK-04 D-60)
  - ``backends/_orchestrator_daemon.py`` — the ONLY direct process-control module (BCK-04)
  - ``viz/server.py``                    — owns its OWN viz_server.pid lifecycle;
                                           its ``os.kill`` calls are a daemon-liveness
                                           probe (``os.kill(pid, 0)``) and SIGTERM stop,
                                           NOT job-control.  Same category as
                                           ``_orchestrator_daemon.py``'s own PID-file
                                           management. (Plan-check iter 1 B-04: lines
                                           235/301/316 of viz/server.py would have failed
                                           the lint script on first run had this allowlist
                                           not been extended; migrating viz daemon PID
                                           lifecycle to a backend would be Phase 7
                                           hardware-autodetect scope creep.)
  - ``cli/cancel.py``                    — OPS-01 (ISSUE-011, v1.1 Phase 13): the CLI
                                           cancel command signals a daemon-launched LOCAL
                                           job's process group DIRECTLY (``os.kill(pid, 0)``
                                           liveness probe + ``os.killpg`` SIGTERM→grace→
                                           SIGKILL) reading pid/pgid from the on-disk
                                           running spec. A fresh CLI process CANNOT route
                                           through the daemon's ``_kill_experiment`` — that
                                           reads the daemon's in-memory ``self.running``
                                           map, which is empty in a separate process — so
                                           direct signalling is the only correct path.
                                           Sanctioned LOCAL-backend job-control, PID-reuse-
                                           guarded by the same ``starttime_ticks`` cross-
                                           check ``_orchestrator_daemon.py`` uses. Same
                                           process-control category as the daemon itself.

Usage:
    uv run python scripts/check_backend_isolation.py [src_root]

    src_root defaults to ``src/automil`` (relative to cwd).

Exit codes:
    0 — no violations
    1 — violations found (file:line diagnostics on stderr)
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Forbidden patterns
# ---------------------------------------------------------------------------

# Bare Name references (e.g., ``Popen(...)``, or an alias that resolves to Popen).
FORBIDDEN_NAMES: frozenset[str] = frozenset({"Popen"})

# Attribute accesses on ``os`` (e.g., ``os.kill``, ``os.killpg``, ``os.getpid``).
FORBIDDEN_OS_ATTRS: frozenset[str] = frozenset({"kill", "killpg", "getpid"})

# Exact attribute name that is forbidden: ``.pid`` (e.g., ``process.pid``).
# Note: ``pid_file``, ``pid_path``, etc. are NOT flagged — only bare ``.pid``.
FORBIDDEN_ATTR: str = "pid"

# ---------------------------------------------------------------------------
# Allowlist (relative to src_root)
# ---------------------------------------------------------------------------

ALLOWLIST_PATHS: frozenset[Path] = frozenset({
    Path("backends/local.py"),
    Path("backends/_orchestrator_daemon.py"),
    Path("backends/pidfile.py"),  # PID-file helpers extracted from the daemon (public liveness surface)
    Path("viz/server.py"),
    Path("cli/cancel.py"),  # OPS-01 CLI direct-kill of daemon-launched local jobs (see docstring rationale)
})


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class BackendIsolationVisitor(ast.NodeVisitor):
    """Walk a module AST and collect BCK-04 violations.

    Detects:
    1. ``os.kill``, ``os.killpg``, ``os.getpid`` attribute accesses.
    2. ``subprocess.Popen`` attribute access or bare ``Popen`` name.
    3. ``from os import kill`` / ``from subprocess import Popen`` (direct or aliased).
    4. ``from os import *`` / ``from subprocess import *`` star-imports (Pitfall 4).
    5. ``.pid`` attribute access (e.g., ``process.pid``).

    False-positive note: ``.pid`` is broad — it catches ``process.pid``,
    ``exp.process.pid``, etc.  It does NOT flag attribute names like
    ``pid_file`` or ``pid_path`` because the check is ``node.attr == "pid"``
    (exact match), not a prefix/startswith check.  Current codebase has no
    ``.pid`` accesses outside the allowlisted modules (RESEARCH.md §4 verified).
    """

    def __init__(self, file_path: Path, src_root: Path) -> None:
        self.file_path = file_path
        self.src_root = src_root
        self.violations: list[tuple[int, str]] = []
        # local_name → original_name, built from import statements.
        self._alias_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Import tracking
    # ------------------------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track ``from os import kill as k`` and star-imports (Pitfall 4)."""
        module = node.module or ""
        if module in ("os", "subprocess"):
            for alias in node.names:
                if alias.name == "*":
                    # Star-import of os/subprocess outside allowlist is forbidden.
                    self.violations.append(
                        (node.lineno, f"forbidden: star-import of {module}")
                    )
                else:
                    # Track the local name so visit_Name catches aliased uses.
                    local = alias.asname or alias.name
                    self._alias_map[local] = alias.name
                    # Eagerly flag at import site too if name is forbidden.
                    if alias.name in FORBIDDEN_NAMES:
                        self.violations.append(
                            (node.lineno,
                             f"forbidden: from {module} import {alias.name}")
                        )
                    if module == "os" and alias.name in FORBIDDEN_OS_ATTRS:
                        self.violations.append(
                            (node.lineno,
                             f"forbidden: from os import {alias.name}")
                        )
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Name references
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        """Catch bare ``Popen`` and aliased forbidden names."""
        # Check the name directly.
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(
                (node.lineno, f"forbidden name: {node.id!r}")
            )
        # Check if this name is an alias for a forbidden original.
        original = self._alias_map.get(node.id)
        if original and original in FORBIDDEN_NAMES:
            self.violations.append(
                (node.lineno,
                 f"forbidden aliased name: {node.id!r} (aliases {original!r})")
            )
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Attribute accesses
    # ------------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Catch ``os.kill``, ``os.killpg``, ``os.getpid``, ``subprocess.Popen``,
        and bare ``.pid`` attribute access.
        """
        if isinstance(node.value, ast.Name):
            root = node.value.id
            # os.kill, os.killpg, os.getpid
            if root == "os" and node.attr in FORBIDDEN_OS_ATTRS:
                self.violations.append(
                    (node.lineno, f"forbidden: os.{node.attr}")
                )
            # subprocess.Popen
            if root == "subprocess" and node.attr in FORBIDDEN_NAMES:
                self.violations.append(
                    (node.lineno, f"forbidden: subprocess.{node.attr}")
                )

        # .pid attribute (exact match — does NOT flag pid_file, pid_path, etc.)
        if node.attr == FORBIDDEN_ATTR:
            self.violations.append(
                (node.lineno, f"forbidden: .{FORBIDDEN_ATTR} attribute access")
            )

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Walk src_root and report BCK-04 violations.

    Returns:
        0 if no violations; 1 otherwise.
    """
    args = argv if argv is not None else sys.argv[1:]
    src_root = Path(args[0]) if args else Path("src/automil")

    if not src_root.is_dir():
        print(f"ERROR: src_root {src_root!r} is not a directory.", file=sys.stderr)
        return 2

    violations: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        rel = py_file.relative_to(src_root)
        if rel in ALLOWLIST_PATHS:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            violations.append(f"{py_file}:0: SyntaxError: {exc}")
            continue
        except OSError as exc:
            violations.append(f"{py_file}:0: IOError: {exc}")
            continue

        visitor = BackendIsolationVisitor(py_file, src_root)
        visitor.visit(tree)
        for lineno, msg in visitor.violations:
            violations.append(f"{py_file}:{lineno}: {msg}")

    if violations:
        print("BCK-04 VIOLATIONS:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print("OK: no backend isolation violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Submit-time smoke run of a variant module through a consumer-owned command.

The framework never imports agent code and knows nothing about the training
seams, so the consumer declares the command (``registry.policy_smoke`` in
``config.yaml``) and the framework runs it in a subprocess with a timeout,
refusing the submission on a non-zero exit or a timeout. ``{python}`` in the
command expands to the interpreter running ``automil``; ``{module}`` to the
module's absolute path, which is otherwise appended as the last argument.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from automil.registry.config import PolicySmoke
from automil.registry.errors import ValidationError

_TAIL_CHARS = 2000


def smoke_argv(smoke: PolicySmoke, module_path: Path) -> list[str]:
    argv = [
        str(module_path) if token == "{module}"
        else sys.executable if token == "{python}"
        else token
        for token in smoke.command
    ]
    if "{module}" not in smoke.command:
        argv.append(str(module_path))
    return argv


def run_policy_smoke(module_path: Path, smoke: PolicySmoke, *, cwd: Path) -> None:
    """Run the declared smoke command on ``module_path``; raise
    :class:`ValidationError` (validator ``smoke``) when it fails or hangs."""
    argv = smoke_argv(smoke, module_path.resolve())
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True,
            timeout=smoke.timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(
            validator_name="smoke", path=module_path,
            reason=f"smoke run timed out after {smoke.timeout_s}s ({' '.join(argv[:-1])}).",
            fix_suggestion="a policy must complete a few optimizer steps on a tiny "
                           "model in seconds; remove the blocking work.",
        ) from exc
    except OSError as exc:
        raise ValidationError(
            validator_name="smoke", path=module_path,
            reason=f"smoke command could not start: {exc}",
            fix_suggestion="check registry.policy_smoke.command in automil/config.yaml.",
        ) from exc
    if proc.returncode == 0:
        return
    tail = (proc.stderr or proc.stdout or "").strip()[-_TAIL_CHARS:]
    raise ValidationError(
        validator_name="smoke", path=module_path,
        reason=f"smoke run exited {proc.returncode}: {tail or 'no output'}",
        fix_suggestion="read the diagnostics above; the policy failed on a seam the "
                       "trainers use, so an attempt would have crashed the same way.",
    )

"""Internal helpers shared across automil CLI subcommands.

Private to the cli/ package (D-02). If the registry or backends layer needs
git-root lookup in Phase 1+, lift to ``automil/paths.py`` at that point — not
now.
"""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Module-level project override set by the --project group option on `main`.
# None = use cwd walk (default behaviour).
# Path = use this path for project discovery (set by main() callback before any subcommand).
# Tests must reset this to None in teardown via monkeypatch.setattr to prevent bleed.
_PROJECT_OVERRIDE: Path | None = None


def _find_automil_dir() -> Path:
    """Walk up from cwd to find a directory containing automil/config.yaml.

    Honors _PROJECT_OVERRIDE when set by the --project group option before
    falling through to the cwd walk.

    Returns the ``automil/`` directory itself.
    """
    if _PROJECT_OVERRIDE is not None:
        # Accept either: (a) project root (parent of automil/) or (b) automil/ dir itself.
        for candidate in (_PROJECT_OVERRIDE / "automil", _PROJECT_OVERRIDE):
            if (candidate / "config.yaml").exists():
                # Always normalise: return the automil/ dir.
                return candidate if candidate.name == "automil" else candidate / "automil"
        raise click.ClickException(
            f"--project {_PROJECT_OVERRIDE}: no automil/config.yaml found under "
            f"{_PROJECT_OVERRIDE}. Point --project at the project root or the "
            f"automil/ dir directly."
        )
    # Existing cwd walk — unchanged:
    p = Path.cwd()
    while p != p.parent:
        candidate = p / "automil" / "config.yaml"
        if candidate.exists():
            return p / "automil"
        p = p.parent
    raise click.ClickException(
        "No automil/config.yaml found. Run 'automil init' in your project root."
    )


def _find_git_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) to find the git repo root."""
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise click.ClickException("Not inside a git repository.")


def _load_technique_map(automil_dir: Path) -> dict[str, str]:
    """Return the consumer's ``scoring.technique_map`` from automil/config.yaml.

    Empty dict on missing config, missing section, malformed type, or any read
    error — the framework default (no auto-extraction) is the safe fall-through.
    Soft-fail with a logged warning rather than aborting the CLI command on
    config drift; the technique_map is a ranking convenience, not a correctness
    contract.

    Schema: ``scoring.technique_map: {pattern: tag, ...}`` where ``pattern`` is
    a literal substring matched against ``description.lower()`` and ``tag`` is
    the technique label written into ``node['techniques']`` when no explicit
    ``--techniques`` was supplied. **Patterns must be lowercase** — the
    description is lowercased before matching but patterns are not, so any
    uppercase character in a pattern guarantees a miss.
    """
    config_path = automil_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse %s for technique_map: %s", config_path, exc)
        return {}
    raw = (cfg.get("scoring") or {}).get("technique_map") or {}
    if not isinstance(raw, dict):
        logger.warning(
            "automil/config.yaml: scoring.technique_map must be a mapping "
            "(pattern -> tag); got %s. Falling back to empty map.",
            type(raw).__name__,
        )
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            logger.warning(
                "scoring.technique_map: entry %r -> %r must be str -> str; "
                "skipping.", k, v,
            )
            continue
        out[k] = v
    return out


def _matches_scope(path: str, patterns: list[str] | set[str]) -> bool:
    """Return whether a relative path matches any configured scope pattern.

    Supports exact file paths, directory prefixes ending in ``/``, and glob
    patterns such as ``data/*.py``.
    """
    rel_path = Path(path).as_posix()
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip().replace("\\", "/")
        if not pattern:
            continue
        if pattern.endswith("/"):
            if rel_path.startswith(pattern):
                return True
            continue
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False

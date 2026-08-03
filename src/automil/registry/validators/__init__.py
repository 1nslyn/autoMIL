"""Submit-time + instantiate-time validators for variant modules (REG-03 / D-30).

Phase 1 ships:
  - InterfaceValidator (Plan 01-04) — submit-time, fully static AST checks.
  - PurityValidator   (Plan 01-04) — submit-time, pure AST.
  - IdentityValidator (Plan 01-05) — instantiate-time, runtime.

The submit hook (Plan 01-07) runs purity FIRST, then interface, short-circuit
on first failure (T-01-14 mitigation: purity catches malicious top-level calls
before interface ever imports the module).
"""
from __future__ import annotations

from automil.registry.validators.identity import IdentityValidator
from automil.registry.validators.interface import InterfaceValidator
from automil.registry.validators.purity import PurityValidator

__all__ = ["InterfaceValidator", "PurityValidator", "IdentityValidator"]

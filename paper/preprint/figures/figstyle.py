"""Shared matplotlib style for every preprint figure.

**All figure text is Times New Roman.** Standing requirement for this project
(2026-07-30): axis labels, tick labels, titles, annotations, legends, colorbar
labels and in-cell numbers all render in Times New Roman, so a figure dropped
into a serif manuscript does not read as visibly foreign.

Every figure script imports :func:`apply` rather than setting ``font.family``
itself -- three scripts each carrying their own hardcoded family is how one of
them silently drifts back to the sans-serif default.

Fallback chain notes (verified on macOS 2026-07-30 via
``matplotlib.font_manager``): ``Times New Roman`` and ``Times`` are present;
``Nimbus Roman`` and ``Liberation Serif`` are NOT, so they are deliberately
absent from the chain. ``STIX Two Text`` and ``DejaVu Serif`` are the
last-resort entries for machines without the Microsoft face -- both are serif,
so the worst case is still a serif figure, never a sans-serif one.

``mathtext.fontset = "stix"`` is load-bearing: without it any ``$...$``
expression renders in DejaVu and the mismatch beside Times body text is
obvious.
"""

from __future__ import annotations

import matplotlib as mpl

#: Preferred serif stack, most-preferred first. See module docstring for why
#: Nimbus Roman / Liberation Serif are excluded.
SERIF_STACK = ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"]

#: Font family used where fixed-width alignment is wanted (numeric table
#: columns). Still Times New Roman -- the requirement admits no mono exception.
TABLE_FAMILY = "Times New Roman"

#: rcParams every figure shares. Scripts may update() further on top.
BASE_RC = {
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "mathtext.fontset": "stix",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def apply(**overrides) -> None:
    """Install the shared style, then any per-script ``overrides``.

    Call once at module import time in each figure script, before any figure is
    created -- rcParams are read when artists are constructed, so applying this
    after a ``plt.subplots()`` leaves that figure on the old font.
    """
    mpl.rcParams.update({**BASE_RC, **overrides})


def resolved_family() -> str:
    """The serif face matplotlib will actually use, for logging/verification.

    Returns the first entry of :data:`SERIF_STACK` that is registered with the
    font manager, or a ``"MISSING:"``-prefixed marker if none are -- so a script
    can print what it got instead of silently emitting the wrong face.
    """
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in SERIF_STACK:
        if name in available:
            return name
    return f"MISSING:none of {SERIF_STACK} registered"

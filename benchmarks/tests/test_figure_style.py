"""Every preprint figure must render its text in Times New Roman.

Standing project requirement (2026-07-30). It is easy to satisfy once and lose
later: three figure scripts each used to carry their own hardcoded
``"font.family": "DejaVu Sans"``, so a fourth script -- or a careless revert --
silently reintroduces a sans-serif figure. These tests pin the rule at the
source (``figstyle``), assert no script sets a family of its own, and check the
end-to-end PNG path actually resolves to Times New Roman.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

pytest.importorskip("matplotlib")

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper", "preprint", "figures",
)
sys.path.insert(0, FIGURES_DIR)

import figstyle  # noqa: E402

#: Every figure-producing script in the preprint.
FIGURE_SCRIPTS = ("make_figures.py", "make_mock_figures.py", "make_dataset_table.py")


class TestFigstyle:
    def test_times_new_roman_is_first_choice(self):
        assert figstyle.SERIF_STACK[0] == "Times New Roman"

    def test_family_is_serif_not_sans(self):
        assert figstyle.BASE_RC["font.family"] == "serif"

    def test_no_sans_serif_anywhere_in_the_stack(self):
        joined = " ".join(figstyle.SERIF_STACK).lower()
        assert "sans" not in joined
        assert "dejavu sans" not in joined

    def test_mathtext_does_not_fall_back_to_dejavu(self):
        """Without stix, ``$...$`` renders in DejaVu beside Times body text."""
        assert figstyle.BASE_RC["mathtext.fontset"] == "stix"

    def test_table_family_is_times_new_roman(self):
        """The rule admits no monospace exception for numeric table columns."""
        assert figstyle.TABLE_FAMILY == "Times New Roman"

    def test_unavailable_faces_stay_out_of_the_stack(self):
        """Nimbus Roman / Liberation Serif are not installed on the target box."""
        assert "Nimbus Roman" not in figstyle.SERIF_STACK
        assert "Liberation Serif" not in figstyle.SERIF_STACK

    def test_apply_sets_rcparams(self):
        import matplotlib as mpl

        figstyle.apply(**{"font.size": 7})
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.serif"][0] == "Times New Roman"
        assert mpl.rcParams["font.size"] == 7

    def test_apply_overrides_do_not_clobber_the_font(self):
        import matplotlib as mpl

        figstyle.apply(**{"axes.labelsize": 11})
        assert mpl.rcParams["font.serif"][0] == "Times New Roman"

    def test_resolved_family_reports_a_real_face(self):
        resolved = figstyle.resolved_family()
        assert not resolved.startswith("MISSING:"), resolved
        assert resolved in figstyle.SERIF_STACK


class TestScriptsDelegateToFigstyle:
    @pytest.mark.parametrize("script", FIGURE_SCRIPTS)
    def test_script_does_not_hardcode_a_font_family(self, script):
        with open(os.path.join(FIGURES_DIR, script)) as f:
            source = f.read()
        # Strip comments so the explanatory notes about the old default do not
        # trip the check.
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        offenders = re.findall(r'"font\.family"\s*:\s*"([^"]+)"', code)
        assert not offenders, (
            f"{script} sets font.family directly ({offenders}); "
            "call figstyle.apply() instead so the family has one source of truth"
        )
        assert "DejaVu Sans" not in code, f"{script} still references DejaVu Sans"

    @pytest.mark.parametrize("script", FIGURE_SCRIPTS)
    def test_script_imports_figstyle(self, script):
        with open(os.path.join(FIGURES_DIR, script)) as f:
            source = f.read()
        assert "figstyle" in source, f"{script} does not use the shared figure style"


class TestRenderedOutput:
    def test_rendered_text_uses_times_new_roman(self, tmp_path):
        """End-to-end: the face matplotlib picks for real text is Times New Roman."""
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        figstyle.apply()
        fig, ax = plt.subplots()
        label = ax.set_xlabel("c-index")
        fig.canvas.draw()
        resolved = fm.findfont(label.get_fontproperties())
        plt.close(fig)
        assert "times" in os.path.basename(resolved).lower(), resolved

    def test_figure_saves_without_font_fallback_warning(self, tmp_path):
        """A missing glyph/face surfaces as a UserWarning -- treat it as failure."""
        import warnings

        import matplotlib.pyplot as plt

        figstyle.apply()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fig, ax = plt.subplots()
            ax.set_title("Survival OS c-index by dataset × arm")
            ax.set_ylabel("test c-index (pooled per-fold mean ± sd)")
            ax.plot([0, 1], [0.5, 0.6])
            out = tmp_path / "probe.png"
            fig.savefig(out)
            plt.close(fig)
        assert out.stat().st_size > 0

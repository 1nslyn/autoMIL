"""H-5b (audit 2026-07-23): no multiple-comparison control anywhere.

The only correction in the repo was ``src/automil/gate/stats.py::bonferroni_correct``,
applied to ``K_effective`` held-out cells inside a *single* gate decision
(default 2). Across the grid -- and across the per-cell lifts the headline now
rests on -- nothing was corrected at all.

Expected values below are hand-computed and cross-checked against R's
``p.adjust`` semantics. Both procedures have tie/monotonicity rules that are
easy to get subtly wrong, so each is pinned by an explicit vector:

  Holm (step-down, cumulative max):
      p.adjust(c(0.01, 0.04, 0.03), "holm") == c(0.03, 0.06, 0.06)
  BH (step-up, reverse cumulative min):
      p.adjust(c(0.01, 0.04, 0.03), "BH")   == c(0.03, 0.04, 0.04)

Note how the same input separates them: Holm pushes 0.04's naive 0.04 *up* to
0.06, while BH pulls 0.03's naive 0.045 *down* to 0.04 because a larger
p-value later in the sequence has a smaller adjusted value.
"""
from __future__ import annotations

import pytest

from autobench.stats import adjust, benjamini_hochberg, holm_bonferroni


def _adj(result):
    return [v["p_adjusted"] for v in result.values()]


def _rej(result):
    return [v["reject"] for v in result.values()]


class TestHolmBonferroni:
    def test_step_down_monotonicity(self):
        """A smaller raw p later in the sort order must drag its successors up."""
        res = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
        assert _adj(res) == pytest.approx([0.03, 0.06, 0.06])

    def test_ties_receive_identical_adjusted_values(self):
        res = holm_bonferroni({"a": 0.01, "b": 0.01, "c": 0.04})
        assert _adj(res) == pytest.approx([0.03, 0.03, 0.04])

    def test_smallest_p_gets_the_full_bonferroni_multiplier(self):
        res = holm_bonferroni({"a": 0.004, "b": 0.9, "c": 0.9, "d": 0.9, "e": 0.9})
        assert res["a"]["p_adjusted"] == pytest.approx(5 * 0.004)

    def test_adjusted_values_are_capped_at_one(self):
        res = holm_bonferroni({"a": 0.5, "b": 0.6})
        assert _adj(res) == pytest.approx([1.0, 1.0])

    def test_rejection_follows_the_adjusted_value(self):
        res = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
        assert _rej(res) == [True, False, False]

    def test_single_comparison_is_uncorrected(self):
        res = holm_bonferroni({"only": 0.02})
        assert res["only"]["p_adjusted"] == pytest.approx(0.02)
        assert res["only"]["reject"] is True


class TestBenjaminiHochberg:
    def test_step_up_monotonicity_pulls_an_earlier_value_down(self):
        """The classic BH case: 0.03's naive 3/2*0.03 = 0.045 is forced down to
        0.04 by the larger p-value that follows it."""
        res = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
        assert _adj(res) == pytest.approx([0.03, 0.04, 0.04])
        # The naive, un-enforced value would have been 0.045.
        assert res["c"]["p_adjusted"] < 0.045

    def test_ties_receive_identical_adjusted_values(self):
        res = benjamini_hochberg({"a": 0.02, "b": 0.02})
        assert _adj(res) == pytest.approx([0.02, 0.02])

    def test_uniform_ladder_collapses_to_a_single_value(self):
        """p_(i) = i*alpha/m for every i sits exactly on the BH line."""
        res = benjamini_hochberg({
            "a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05,
        })
        assert _adj(res) == pytest.approx([0.05] * 5)

    def test_largest_p_is_never_adjusted(self):
        res = benjamini_hochberg({"a": 0.001, "b": 0.002, "c": 0.4})
        assert res["c"]["p_adjusted"] == pytest.approx(0.4)

    def test_rejection_matches_the_classic_step_up_rule(self):
        """reject <=> p_adjusted <= alpha must equal the textbook rule
        ``k = max{i : p_(i) <= i*alpha/m}, reject H_(1..k)``."""
        p_values = {"a": 0.005, "b": 0.02, "c": 0.03, "d": 0.5, "e": 0.9}
        alpha = 0.05
        res = benjamini_hochberg(p_values, alpha=alpha)

        ordered = sorted(p_values.values())
        m = len(ordered)
        k = max(
            (i for i, p in enumerate(ordered, start=1) if p <= i * alpha / m),
            default=0,
        )
        classic_cutoff = ordered[k - 1] if k else -1.0
        expected = {cid: (p <= classic_cutoff) for cid, p in p_values.items()}
        assert {cid: v["reject"] for cid, v in res.items()} == expected

    def test_single_comparison_is_uncorrected(self):
        res = benjamini_hochberg({"only": 0.03})
        assert res["only"]["p_adjusted"] == pytest.approx(0.03)


class TestSharedContract:
    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_input_order_is_preserved(self, fn):
        p_values = {"z": 0.04, "a": 0.01, "m": 0.03}
        assert list(fn(p_values).keys()) == ["z", "a", "m"]

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_raw_p_value_is_echoed_back(self, fn):
        res = fn({"a": 0.01, "b": 0.04})
        assert res["a"]["p_value"] == pytest.approx(0.01)
        assert res["b"]["p_value"] == pytest.approx(0.04)

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_adjusted_is_never_below_raw(self, fn):
        p_values = {f"c{i}": p for i, p in enumerate([0.001, 0.01, 0.02, 0.3, 0.7])}
        res = fn(p_values)
        for cid, entry in res.items():
            assert entry["p_adjusted"] >= entry["p_value"] - 1e-15

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_adjusted_is_monotone_in_the_raw_ordering(self, fn):
        p_values = {f"c{i}": p for i, p in enumerate([0.001, 0.01, 0.02, 0.3, 0.7])}
        res = fn(p_values)
        adjusted = [res[cid]["p_adjusted"] for cid in sorted(res, key=lambda c: res[c]["p_value"])]
        assert adjusted == sorted(adjusted)

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_empty_family_is_a_no_op(self, fn):
        assert fn({}) == {}

    def test_holm_never_rejects_more_than_bh(self):
        """FWER control is strictly more conservative than FDR control."""
        p_values = {f"c{i}": p for i, p in enumerate(
            [0.001, 0.008, 0.02, 0.03, 0.04, 0.2, 0.5, 0.9]
        )}
        holm = {c for c, v in holm_bonferroni(p_values).items() if v["reject"]}
        bh = {c for c, v in benjamini_hochberg(p_values).items() if v["reject"]}
        assert holm <= bh

    def test_alpha_is_honoured(self):
        p_values = {"a": 0.01, "b": 0.04, "c": 0.03}
        strict = holm_bonferroni(p_values, alpha=0.01)
        loose = holm_bonferroni(p_values, alpha=0.10)
        assert _rej(strict) == [False, False, False]
        assert _rej(loose) == [True, True, True]


class TestInputValidation:
    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_nan_p_value_names_the_offender(self, fn):
        """Silently dropping a NaN would shrink m and weaken the correction --
        the same class of hidden-denominator defect as M-15."""
        with pytest.raises(ValueError, match="cell_luad_kras"):
            fn({"cell_luad_kras": float("nan"), "b": 0.01})

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    @pytest.mark.parametrize("bad", [-0.01, 1.5, float("inf")])
    def test_out_of_range_p_value_rejected(self, fn, bad):
        with pytest.raises(ValueError, match="offending"):
            fn({"offending": bad, "b": 0.01})

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.2])
    def test_alpha_out_of_range_rejected(self, fn, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            fn({"a": 0.01}, alpha=bad_alpha)

    @pytest.mark.parametrize("fn", [holm_bonferroni, benjamini_hochberg])
    def test_non_numeric_p_value_rejected(self, fn):
        with pytest.raises(ValueError, match="bad_id"):
            fn({"bad_id": "0.01"})


class TestDispatcher:
    def test_holm_dispatch_matches_direct_call(self):
        p_values = {"a": 0.01, "b": 0.04, "c": 0.03}
        assert adjust(p_values, method="holm") == holm_bonferroni(p_values)

    def test_bh_dispatch_matches_direct_call(self):
        p_values = {"a": 0.01, "b": 0.04, "c": 0.03}
        assert adjust(p_values, method="bh") == benjamini_hochberg(p_values)

    def test_default_method_is_holm(self):
        """The recommended family control for the headline per-cell lift table."""
        p_values = {"a": 0.01, "b": 0.04, "c": 0.03}
        assert adjust(p_values) == holm_bonferroni(p_values)

    def test_unknown_method_fails_fast(self):
        with pytest.raises(ValueError, match="method"):
            adjust({"a": 0.01}, method="bonferroni")

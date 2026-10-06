"""Protocol v5: the winner rule (campaign.max_lift_winner)."""
from __future__ import annotations

import json
import math
import random
import statistics

import pytest

from autobench.campaign import (
    DISCOVERY_ATTEMPTS,
    WINNER_FLOOR,
    WINNER_SE_MULTIPLIER,
    CampaignManifestError,
    max_lift_winner,
)

BASELINE = [0.70, 0.72, 0.68, 0.71, 0.69]


def _candidate(name, lifts):
    return {
        "candidate_id": name,
        "candidate_sha256": name * 8,
        "fold_values": [base + lift for base, lift in zip(BASELINE, lifts)],
    }


def test_an_empty_pool_keeps_the_baseline():
    record = max_lift_winner(BASELINE, [])
    assert record["winner"] == "baseline"
    assert record["accepted"] is False
    assert record["pool"] == [] and record["leader"] is None
    assert record["baseline_fold_values"] == BASELINE


def test_the_numbers_match_a_direct_computation():
    pool = [
        _candidate("a", [0.05, 0.03, 0.06, 0.04, 0.05]),
        _candidate("b", [0.01, -0.01, 0.02, 0.00, 0.01]),
        _candidate("c", [-0.02, 0.00, -0.01, -0.03, -0.01]),
    ]
    record = max_lift_winner(BASELINE, pool)
    deltas = [
        [value - base for value, base in zip(c["fold_values"], BASELINE)] for c in pool
    ]
    fold_variance = statistics.fmean(statistics.variance(row) for row in deltas)
    standard_error = math.sqrt(fold_variance / len(BASELINE))
    bar = max(WINNER_FLOOR, WINNER_SE_MULTIPLIER * standard_error)
    assert record["leader"] == "a"
    assert record["leader_lift"] == pytest.approx(statistics.fmean(deltas[0]), abs=1e-12)
    assert record["fold_variance"] == pytest.approx(fold_variance, abs=1e-12)
    assert record["standard_error"] == pytest.approx(standard_error, abs=1e-12)
    assert record["bar"] == pytest.approx(bar, abs=1e-12)
    assert record["accepted"] is (record["leader_lift"] > record["bar"]) is True
    assert record["winner"] == "a"


def test_the_bar_is_set_for_the_best_of_every_attempt_a_cell_launches():
    """2.93 is the one-sided 5% Sidak critical value for the maximum of the
    cell's 30 attempts, rounded up."""
    sidak = statistics.NormalDist().inv_cdf(0.95 ** (1 / DISCOVERY_ATTEMPTS))
    assert sidak <= WINNER_SE_MULTIPLIER < sidak + 0.01


def test_a_lead_inside_the_noise_of_the_search_keeps_the_baseline():
    # 0.03 would clear two standard errors for one attempt, not the bar for
    # the best of thirty.
    pool = [
        _candidate("a", [0.06, 0.00, 0.06, 0.00, 0.03]),
        _candidate("b", [0.04, -0.02, 0.04, -0.02, 0.01]),
    ]
    record = max_lift_winner(BASELINE, pool)
    assert record["leader"] == "a"
    assert 2 * record["standard_error"] < record["leader_lift"] < record["bar"]
    assert record["winner"] == "baseline"


def test_the_bar_does_not_move_with_how_bad_the_other_attempts_are():
    leader = _candidate("lead", [0.06, 0.01, 0.05, 0.02, 0.04])
    near = [_candidate(f"n{i}", [0.02, -0.03, 0.01, -0.02, 0.00]) for i in range(5)]
    far = [_candidate(f"n{i}", [-0.13, -0.18, -0.14, -0.17, -0.15]) for i in range(5)]
    with_near = max_lift_winner(BASELINE, [leader, *near])
    with_far = max_lift_winner(BASELINE, [leader, *far])
    assert with_near["bar"] == pytest.approx(with_far["bar"], abs=1e-15)
    assert with_near["accepted"] is with_far["accepted"] is True


def test_a_single_candidate_is_judged_on_its_own_lift():
    pool = [_candidate("only", [0.03, 0.03, 0.03, 0.03, 0.03])]
    record = max_lift_winner(BASELINE, pool)
    assert record["leader_lift"] == pytest.approx(0.03, abs=1e-12)
    assert record["standard_error"] == pytest.approx(0.0, abs=1e-12)
    assert record["bar"] == WINNER_FLOOR
    assert record["winner"] == "only"


def test_the_floor_and_the_standard_error_each_can_set_the_bar():
    quiet = max_lift_winner(BASELINE, [_candidate("q", [0.009] * 5)])
    assert quiet["bar"] == WINNER_FLOOR and quiet["winner"] == "baseline"
    noisy = max_lift_winner(
        BASELINE, [_candidate("n", [0.20, -0.10, 0.15, -0.05, 0.10])],
    )
    assert noisy["bar"] == WINNER_SE_MULTIPLIER * noisy["standard_error"] > WINNER_FLOOR
    assert noisy["leader_lift"] < noisy["bar"]
    assert noisy["winner"] == "baseline"


def test_a_lift_exactly_at_the_bar_is_refused():
    record = max_lift_winner([0.0, 0.0], [{
        "candidate_id": "edge", "candidate_sha256": "e" * 64,
        "fold_values": [WINNER_FLOOR, WINNER_FLOOR],
    }])
    assert record["leader_lift"] == record["bar"] == WINNER_FLOOR
    assert record["winner"] == "baseline"


def test_tied_leaders_keep_pool_order():
    lifts = [0.05, 0.05, 0.05, 0.05, 0.05]
    record = max_lift_winner(BASELINE, [
        _candidate("first", lifts), _candidate("second", lifts),
    ])
    assert record["leader"] == "first"


def test_recomputing_from_the_recorded_inputs_gives_the_identical_record():
    pool = [
        _candidate("a", [0.05, 0.03, 0.06, 0.04, 0.05]),
        _candidate("b", [0.01, -0.01, 0.02, 0.00, 0.01]),
    ]
    record = json.loads(json.dumps(max_lift_winner(BASELINE, pool)))
    again = max_lift_winner(record["baseline_fold_values"], record["pool"])
    assert again == record


def test_every_number_comes_from_correctly_rounded_operations():
    """A float ``**`` goes through the platform's pow, which need not round
    a square correctly, so a record built with it would differ between the
    machine that froze it and the one that checks it. Rebuilt from products,
    fsum, division and sqrt alone, every record must come out identical."""
    rng = random.Random(5)
    for _ in range(400):
        baseline = [rng.uniform(0.5, 0.9) for _ in range(5)]
        pool = [{
            "candidate_id": f"c{i}", "candidate_sha256": f"{i:064x}",
            "fold_values": [b + rng.gauss(0.0, 0.03) for b in baseline],
        } for i in range(25)]
        record = max_lift_winner(baseline, pool)
        base_mean = math.fsum(baseline) / 5
        variances = []
        for entry, candidate in zip(record["pool"], pool):
            lift = math.fsum(candidate["fold_values"]) / 5 - base_mean
            residuals = [v - b - lift for v, b in zip(candidate["fold_values"], baseline)]
            variances.append(math.fsum(r * r for r in residuals) / 4)
            assert (entry["lift"], entry["fold_variance"]) == (lift, variances[-1])
        assert record["fold_variance"] == math.fsum(variances) / 25
        assert record["standard_error"] == math.sqrt(record["fold_variance"] / 5)


@pytest.mark.parametrize("baseline, values", [
    ([0.7], [0.8]),
    (BASELINE, [0.8, 0.8, 0.8, 0.8]),
])
def test_unpaired_or_too_few_folds_are_refused(baseline, values):
    with pytest.raises(CampaignManifestError):
        max_lift_winner(baseline, [{
            "candidate_id": "x", "candidate_sha256": "x" * 64, "fold_values": values,
        }])

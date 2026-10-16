"""Protocol v6: the winner rule (campaign.val_best_winner)."""
from __future__ import annotations

import json
import math
import random

import pytest

from autobench.campaign import CampaignManifestError, val_best_winner

BASELINE = [0.70, 0.72, 0.68, 0.71, 0.69]
RECORD_KEYS = {
    "rule", "baseline_fold_values", "pool", "leader", "leader_lift",
    "accepted", "winner",
}
ENTRY_KEYS = {"candidate_id", "candidate_sha256", "fold_values", "lift"}


def _candidate(name, lifts):
    return {
        "candidate_id": name,
        "candidate_sha256": name * 8,
        "fold_values": [base + lift for base, lift in zip(BASELINE, lifts)],
    }


def test_an_empty_pool_keeps_the_baseline():
    record = val_best_winner(BASELINE, [])
    assert record["winner"] == "baseline"
    assert record["accepted"] is False
    assert record["pool"] == []
    assert record["leader"] is None and record["leader_lift"] is None
    assert record["baseline_fold_values"] == BASELINE


def test_the_record_holds_exactly_the_rule_keys():
    pool = [_candidate("a", [0.05, 0.03, 0.06, 0.04, 0.05])]
    for record in (val_best_winner(BASELINE, []), val_best_winner(BASELINE, pool)):
        assert set(record) == RECORD_KEYS
        assert record["rule"] == "val-best"
        assert all(set(entry) == ENTRY_KEYS for entry in record["pool"])


def test_the_leader_with_the_highest_lift_is_deployed():
    pool = [
        _candidate("a", [0.05, 0.03, 0.06, 0.04, 0.05]),
        _candidate("b", [0.01, -0.01, 0.02, 0.00, 0.01]),
        _candidate("c", [-0.02, 0.00, -0.01, -0.03, -0.01]),
    ]
    record = val_best_winner(BASELINE, pool)
    assert record["leader"] == "a"
    assert record["leader_lift"] == pytest.approx(0.046, abs=1e-12)
    assert [entry["candidate_id"] for entry in record["pool"]] == ["a", "b", "c"]
    assert record["pool"][1]["lift"] == pytest.approx(0.006, abs=1e-12)
    assert record["pool"][1]["candidate_sha256"] == "bbbbbbbb"
    assert record["accepted"] is True
    assert record["winner"] == "a"


def test_a_tiny_positive_leader_is_deployed():
    record = val_best_winner(BASELINE, [_candidate("a", [0.001] * 5)])
    assert record["leader_lift"] == pytest.approx(0.001, abs=1e-12)
    assert record["accepted"] is True and record["winner"] == "a"


def test_a_leader_inside_the_fold_noise_is_still_deployed():
    """No significance gate: a positive mean lift with per-fold deltas that
    flip sign still wins."""
    noisy = _candidate("noisy", [0.20, -0.10, 0.15, -0.05, 0.10])
    record = val_best_winner(BASELINE, [noisy])
    assert record["leader_lift"] > 0.0
    assert record["winner"] == "noisy"


def test_a_zero_lift_keeps_the_baseline():
    record = val_best_winner(BASELINE, [_candidate("z", [0.0] * 5)])
    assert record["leader"] == "z" and record["leader_lift"] == 0.0
    assert record["accepted"] is False and record["winner"] == "baseline"


def test_a_negative_leader_keeps_the_baseline():
    pool = [
        _candidate("a", [-0.01, -0.02, -0.01, -0.03, -0.01]),
        _candidate("b", [-0.05, -0.04, -0.05, -0.06, -0.05]),
    ]
    record = val_best_winner(BASELINE, pool)
    assert record["leader"] == "a" and record["leader_lift"] < 0.0
    assert record["accepted"] is False and record["winner"] == "baseline"


def test_tied_leaders_keep_pool_order():
    lifts = [0.05, 0.05, 0.05, 0.05, 0.05]
    record = val_best_winner(BASELINE, [
        _candidate("first", lifts), _candidate("second", lifts),
    ])
    assert record["leader"] == "first" and record["winner"] == "first"
    swapped = val_best_winner(BASELINE, [
        _candidate("second", lifts), _candidate("first", lifts),
    ])
    assert swapped["leader"] == "second"


def test_recomputing_from_the_recorded_inputs_gives_the_identical_record():
    pool = [
        _candidate("a", [0.05, 0.03, 0.06, 0.04, 0.05]),
        _candidate("b", [0.01, -0.01, 0.02, 0.00, 0.01]),
    ]
    record = json.loads(json.dumps(val_best_winner(BASELINE, pool)))
    again = val_best_winner(record["baseline_fold_values"], record["pool"])
    assert again == record


def test_every_number_comes_from_correctly_rounded_operations():
    """The lift is fsum and one division, so the record frozen on one machine
    is the one every other machine recomputes."""
    rng = random.Random(5)
    for _ in range(200):
        baseline = [rng.uniform(0.5, 0.9) for _ in range(5)]
        pool = [{
            "candidate_id": f"c{i}", "candidate_sha256": f"{i:064x}",
            "fold_values": [b + rng.gauss(0.0, 0.03) for b in baseline],
        } for i in range(25)]
        record = val_best_winner(baseline, pool)
        base_mean = math.fsum(baseline) / 5
        for entry, candidate in zip(record["pool"], pool):
            assert entry["lift"] == math.fsum(candidate["fold_values"]) / 5 - base_mean
        best = max(entry["lift"] for entry in record["pool"])
        assert record["leader_lift"] == best
        assert record["accepted"] is (best > 0.0)
        assert record == val_best_winner(baseline, pool)


@pytest.mark.parametrize("baseline, values", [
    ([0.7], [0.8]),
    (BASELINE, [0.8, 0.8, 0.8, 0.8]),
])
def test_unpaired_or_too_few_folds_are_refused(baseline, values):
    with pytest.raises(CampaignManifestError):
        val_best_winner(baseline, [{
            "candidate_id": "x", "candidate_sha256": "x" * 64, "fold_values": values,
        }])


def test_the_old_significance_machinery_is_gone():
    import autobench.campaign as campaign

    for name in ("max_lift_winner", "WINNER_FLOOR", "WINNER_SE_MULTIPLIER"):
        assert not hasattr(campaign, name)

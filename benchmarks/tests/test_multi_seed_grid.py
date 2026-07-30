"""H-5c: the grid must be able to run more than one seed.

``TrainConfig.seed`` is a scalar and the generator never looped over it, so the
whole campaign was one seed × K folds. The reported cross-fold ``std`` therefore
captures **partition variance under a single fixed partition only** — it contains
zero training-stochasticity component (initialisation, dropout, sampler order,
cuDNN nondeterminism) and cannot be read as run-to-run reproducibility.

That was tolerable while the empirical plan was a static leaderboard. It is not
tolerable for C3: the equal-effort study includes both cross-lineage reranking
and within-lineage lift, and the first question asked of any lift is whether it
exceeds the noise of simply rerunning the same recipe.

The harness is code; **how many seeds to run is Leo's call** — see H-5c in the
tracker. An empty ``seeds`` list reproduces the single-seed grid exactly, so this
changes nothing until somebody asks for it.

Note also what a seed does and does not vary here: splits are generated once and
cached, and the cache is not seed-keyed, so a second seed trains on the SAME
folds from a different initialisation. That is a repeated-measures design and it
is the right one for isolating training stochasticity — but it is not resampling
the partition, and the paper must not describe it as though it were.
"""
from __future__ import annotations

import pytest

from autobench.pipeline.config import (
    BenchmarkConfig,
    Framework,
    build_registries,
    generate_all_experiments,
)
from _helpers import make_test_ds


@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


def _cfg(ds, **kw):
    return BenchmarkConfig.from_dataset_config(ds, **kw)


class TestSingleSeedIsUnchanged:
    """The freeze guard: a config that says nothing about seeds is not affected."""

    def test_seeds_defaults_to_empty(self):
        assert BenchmarkConfig().seeds == []

    def test_an_empty_seed_list_reproduces_the_old_grid(self, ds, registries):
        before = generate_all_experiments(_cfg(ds), registries)
        assert before, "fixture produced no experiments"
        assert {e.train.seed for e in before} == {42}

    def test_every_experiment_id_is_unchanged(self, ds, registries):
        exps = generate_all_experiments(_cfg(ds), registries)
        assert all(e.experiment_id.endswith("__s42")
                   or "__s42__" in e.experiment_id for e in exps)


class TestSeedAxis:
    def test_three_seeds_triple_the_grid(self, ds, registries):
        one = generate_all_experiments(_cfg(ds), registries)
        three = generate_all_experiments(_cfg(ds, seeds=[42, 43, 44]), registries)
        assert len(three) == 3 * len(one)

    def test_each_seed_is_present(self, ds, registries):
        exps = generate_all_experiments(_cfg(ds, seeds=[42, 43, 44]), registries)
        assert {e.train.seed for e in exps} == {42, 43, 44}

    def test_seeds_do_not_collide_on_experiment_id(self, ds, registries):
        exps = generate_all_experiments(_cfg(ds, seeds=[42, 43, 44]), registries)
        assert len({e.experiment_id for e in exps}) == len(exps)

    def test_seeds_do_not_collide_on_the_results_path(self, ds, registries):
        """CR-5b is what makes this safe: without the seed segment, seed 43 would
        resume seed 42's per-fold metrics.json and report zero variance."""
        exps = generate_all_experiments(_cfg(ds, seeds=[42, 43]), registries)
        assert len({e.results_subdir for e in exps}) == len(exps)

    def test_a_single_element_seed_list_equals_the_default_grid(self, ds, registries):
        default = generate_all_experiments(_cfg(ds), registries)
        explicit = generate_all_experiments(_cfg(ds, seeds=[42]), registries)
        assert ({e.experiment_id for e in default}
                == {e.experiment_id for e in explicit})

    def test_a_seed_list_without_the_default_replaces_it(self, ds, registries):
        exps = generate_all_experiments(_cfg(ds, seeds=[7, 8]), registries)
        assert {e.train.seed for e in exps} == {7, 8}

    def test_everything_but_the_seed_is_held_constant(self, ds, registries):
        """A repeated-measures design: the ONLY thing that varies is the seed."""
        exps = generate_all_experiments(_cfg(ds, seeds=[42, 43]), registries)
        by_seed = {}
        for e in exps:
            by_seed.setdefault(e.train.seed, []).append(
                (e.framework.value, e.strategy, e.task.name, e.encoder_key,
                 e.model.model_type, e.survival_loss, e.n_folds)
            )
        assert sorted(by_seed[42]) == sorted(by_seed[43])

    def test_the_shared_train_config_is_not_mutated(self, ds, registries):
        """Immutability: replace(), not an in-place seed assignment — otherwise
        every experiment would end up carrying the last seed in the list."""
        cfg = _cfg(ds, seeds=[42, 43, 44])
        generate_all_experiments(cfg, registries)
        assert cfg.train.seed == 42

    def test_other_train_fields_survive_the_seed_swap(self, ds, registries):
        from dataclasses import replace as dc_replace
        from autobench.pipeline.config import TrainConfig

        cfg = _cfg(ds, seeds=[42, 43])
        cfg.train = dc_replace(TrainConfig(), lr=7e-4, max_epochs=13)
        exps = generate_all_experiments(cfg, registries)
        assert all(e.train.lr == 7e-4 and e.train.max_epochs == 13 for e in exps)

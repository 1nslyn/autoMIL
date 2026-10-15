"""The composite recipe policy, end to end.

``benchmarks/experiments/dev/_recipes/composite.py`` is one ``Recipe`` policy whose
switches are class constants; ``dev_recipe.py`` bakes a set of constants into a copy
of it and checks the copy the way ``automil submit`` does. Here that file

* instantiates with its constants baked in and passes the sealed-policy validators;
* passes the policy smoke on every arm: every switch off, every switch the arm takes on
  together, and each of those alone (nnMIL refuses the tile caps, the others the body scale);
* reproduces a native run exactly with every switch off, on every arm's real trainer;
* changes the run when a switch is on, on every arm the switch applies to.

``test_recipe_units.py`` (the optimizer wrapper) and ``test_recipe_seams.py`` (the other
seams) hold the closed-form checks of the machinery itself.
"""
from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from _recipe_probe import (  # noqa: E402
    ALL_ON,
    BASE_COMMIT,
    CREATED_AT,
    NODE_ID,
    SEAL_CONFIG_DIR,
    SWITCHES,
    TEMPLATE,
    all_on,
    captured,
    dev_recipe,
    instantiate,
    recipe_class,
    switches_for,
    unique_name,
    write_recipe,
)
from _seam_probe import fold_runtime  # noqa: E402
from automil.registry._state import SPEC_STORE, _clear_registry  # noqa: E402
from automil.registry.config import load_registry_config  # noqa: E402
from automil.registry.validators import InterfaceValidator, PurityValidator  # noqa: E402
from autobench.pipeline import policy_smoke  # noqa: E402
from autobench.pipeline.policy_dispatch import defines_seam  # noqa: E402
from tests import test_policy_seam_nnmil as nnmil_seams  # noqa: E402
from tests.test_policy_seam_arms import ARMS  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the trainers' GPU is not on this host",
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def _constants(source: str) -> dict[str, object]:
    """The ``NAME = literal`` class constants of the ``Recipe`` class in ``source``."""
    (recipe,) = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Recipe"]
    return {
        stmt.targets[0].id: ast.literal_eval(stmt.value)
        for stmt in recipe.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    }


def _comments(source: str) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [t.string for t in tokens if t.type == tokenize.COMMENT]


# --- instantiating the template ------------------------------------------------------------

MINIMAL = '''"""A template with one constant."""
from automil.registry import PolicyVariant, VariantSpec, register


@register(VariantSpec(
    name="__NAME__", kind="policy", parent=None, base_commit="__BASE_COMMIT__",
    primary_value=0.0, node_id="__NODE_ID__", created_at="__CREATED_AT__",
))
class Recipe(PolicyVariant):
    EMA_DECAY = None  # off
    ARM = "abmil"     # which arm

    def wrap_optimizer(self, opt):
        {body}
'''


def _minimal(tmp_path: Path, body: str = "return opt") -> Path:
    path = tmp_path / "minimal_template.py"
    path.write_text(MINIMAL.format(body=body))
    return path


class TestInstantiate:
    def test_the_constants_are_baked_over_the_templates_defaults(self, tmp_path):
        text = instantiate(EMA_DECAY=0.998, ARM="clam_mb", STEP_ANNEAL=(20, 0.2))
        baked, default = _constants(text), _constants(TEMPLATE.read_text())
        assert baked["EMA_DECAY"] == 0.998 and baked["ARM"] == "clam_mb"
        assert baked["STEP_ANNEAL"] == (20, 0.2)
        assert {k: v for k, v in baked.items() if k not in ("EMA_DECAY", "ARM", "STEP_ANNEAL")} == {
            k: v for k, v in default.items() if k not in ("EMA_DECAY", "ARM", "STEP_ANNEAL")
        }, "constants that were not named keep the template's defaults"

    def test_the_loaded_policy_carries_the_baked_constants_and_its_provenance(self, tmp_path):
        name = unique_name("provenance")
        path = tmp_path / f"{name}.py"
        path.write_text(instantiate(name, LS_EPS=0.1))
        policy = policy_smoke.load_policy_class(path)
        assert policy.LS_EPS == 0.1
        spec = SPEC_STORE[("policy", None, name)]
        assert (spec.node_id, spec.base_commit, spec.created_at) == (NODE_ID, BASE_COMMIT, CREATED_AT)

    def test_lists_become_tuples_at_every_depth(self):
        constants = _constants(instantiate(CLASS_WEIGHTS=[1.0, 2.0], UNFREEZE=[10, 20, 0.1]))
        assert constants["CLASS_WEIGHTS"] == (1.0, 2.0) and isinstance(constants["CLASS_WEIGHTS"], tuple)
        assert constants["UNFREEZE"] == (10, 20, 0.1) and isinstance(constants["UNFREEZE"], tuple)

    def test_every_template_constant_can_be_baked(self):
        template = _constants(TEMPLATE.read_text())
        assert {"ARM", "EMA_DECAY", "LS_EPS", "POSITIVE_SHARE", "CLASS_WEIGHTS", "MAX_TILES", "KEEP_FRAC",
                "WARMUP_STEPS", "WARMUP_START", "COSINE_EPOCHS", "STEP_ANNEAL", "MIN_EPOCHS", "UNFREEZE",
                "BODY_LR_SCALE"} <= set(template)
        baked = _constants(instantiate(**ALL_ON))
        assert all(baked[name] == value for name, value in ALL_ON.items())

    def test_comments_and_the_docstring_survive(self, tmp_path):
        template = _minimal(tmp_path)
        text = dev_recipe().instantiate_recipe(
            template, name="kept", constants={"EMA_DECAY": 0.99}, node_id=NODE_ID,
            base_commit=BASE_COMMIT, created_at=CREATED_AT,
        )
        assert _comments(text) == _comments(template.read_text()) == ["# off", "# which arm"]
        assert ast.get_docstring(ast.parse(text)) == ast.get_docstring(ast.parse(template.read_text()))
        assert _constants(text) == {"EMA_DECAY": 0.99, "ARM": "abmil"}

    def test_an_unknown_constant_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="NO_SUCH_SWITCH"):
            instantiate(NO_SUCH_SWITCH=1)
        with pytest.raises(ValueError, match="wrap_optimizer"):
            instantiate(wrap_optimizer=1)  # not a constant of the template

    @pytest.mark.parametrize("value", [{"a": 1}, {1, 2}, float("nan"), float("inf"), object(), 1j])
    def test_a_value_the_template_cannot_hold_is_refused(self, value):
        with pytest.raises(ValueError, match="EMA_DECAY"):
            instantiate(EMA_DECAY=value)

    @pytest.mark.parametrize("name", ["_private", "", "has space", "quo\"te", "a/b", "new\nline"])
    def test_a_name_must_be_a_plain_name_without_a_leading_underscore(self, name):
        with pytest.raises(ValueError, match="name"):
            instantiate(name)

    @pytest.mark.parametrize("field", ["node_id", "base_commit", "created_at"])
    def test_a_provenance_field_cannot_carry_source_text(self, field):
        fields = {"node_id": NODE_ID, "base_commit": BASE_COMMIT, "created_at": CREATED_AT}
        fields[field] = 'x", primary_value=1.0, node_id="y'
        with pytest.raises(ValueError, match=field):
            dev_recipe().instantiate_recipe(TEMPLATE, name="n", constants={}, **fields)

    def test_the_result_passes_strict_purity_and_the_interface_validator(self, tmp_path):
        path = write_recipe(tmp_path, **ALL_ON)
        seal = load_registry_config(SEAL_CONFIG_DIR).policy_method_imports
        assert seal and "torch" in seal
        PurityValidator(strict_policy=True, method_imports=seal).check(path)
        InterfaceValidator().check(path)

    def test_a_policy_the_validators_refuse_is_refused_with_their_message(self, tmp_path):
        def refused_for(path: Path, message: str) -> None:
            with pytest.raises(ValueError, match=message):
                dev_recipe().instantiate_recipe(
                    path, name="bad", constants={}, node_id=NODE_ID, base_commit=BASE_COMMIT,
                    created_at=CREATED_AT,
                )

        refused_for(_minimal(tmp_path, body="import os\n        return opt"), r"policy methods may import only")
        top = tmp_path / "top.py"
        top.write_text("import os\n" + MINIMAL.format(body="return opt"))
        refused_for(top, "untrusted top-level import")
        # clean under purity, refused by the interface validator: a parameter the ABC does not define
        extra = tmp_path / "extra.py"
        extra.write_text(MINIMAL.format(body="return opt").replace("(self, opt)", "(self, opt, extra)"))
        refused_for(extra, "new required positional parameter")

    @pytest.mark.parametrize("missing", ["__NAME__", "__BASE_COMMIT__", "__NODE_ID__", "__CREATED_AT__"])
    def test_a_template_must_carry_each_placeholder_once(self, tmp_path, missing):
        gone = tmp_path / "gone.py"
        gone.write_text(MINIMAL.format(body="return opt").replace(missing, "fixed"))
        twice = tmp_path / "twice.py"
        twice.write_text(MINIMAL.format(body="return opt").replace("class Recipe", f"# {missing}\nclass Recipe"))
        for template in (gone, twice):
            with pytest.raises(ValueError, match=missing):
                dev_recipe().instantiate_recipe(
                    template, name="n", constants={}, node_id=NODE_ID, base_commit=BASE_COMMIT,
                    created_at=CREATED_AT,
                )

    @pytest.mark.parametrize("field, placeholder", [
        ("name", "__NAME__"), ("base_commit", "__BASE_COMMIT__"),
        ("node_id", "__NODE_ID__"), ("created_at", "__CREATED_AT__"),
    ])
    def test_a_provenance_field_must_reach_the_spec(self, tmp_path, field, placeholder):
        # the placeholder appears once, but in a comment: the spec would keep a fixed value
        lost = tmp_path / "lost.py"
        lost.write_text(
            MINIMAL.format(body="return opt")
            .replace(f'{field}="{placeholder}"', f'{field}="fixed"')
            .replace("class Recipe", f"# {placeholder}\nclass Recipe")
        )
        with pytest.raises(ValueError, match=f"{field} reads back as 'fixed'"):
            dev_recipe().instantiate_recipe(
                lost, name="n", constants={}, node_id=NODE_ID, base_commit=BASE_COMMIT,
                created_at=CREATED_AT,
            )

    def test_a_constant_spread_over_several_lines_is_refused(self, tmp_path):
        spread = tmp_path / "spread.py"
        spread.write_text(MINIMAL.format(body="return opt").replace(
            "EMA_DECAY = None  # off", "EMA_DECAY = (\n        None,\n        None)  # off",
        ))
        with pytest.raises(ValueError, match="EMA_DECAY spans several lines"):
            dev_recipe().instantiate_recipe(
                spread, name="n", constants={"EMA_DECAY": 0.9}, node_id=NODE_ID,
                base_commit=BASE_COMMIT, created_at=CREATED_AT,
            )


class TestInstantiateCommandLine:
    ARGS = ["instantiate", "--node-id", "node_0003", "--base-commit", "abc1234"]

    def test_it_writes_a_policy_with_the_set_constants(self, tmp_path, capsys):
        out, name = tmp_path / "out" / "ema_cos.py", unique_name("cli")
        code = dev_recipe().main([
            *self.ARGS, "--name", name, "--set", "EMA_DECAY=0.998", "--set", "ARM=abmil",
            "--set", "STEP_ANNEAL=[20, 0.2]", "--set", "MIN_EPOCHS=None", "--out", str(out),
        ])
        assert code == 0, capsys.readouterr().err
        constants = _constants(out.read_text())
        assert constants["EMA_DECAY"] == 0.998 and constants["ARM"] == "abmil"
        assert constants["STEP_ANNEAL"] == (20, 0.2) and constants["MIN_EPOCHS"] is None
        spec = policy_smoke.load_policy_class(out)
        assert SPEC_STORE[("policy", None, name)].node_id == "node_0003" and spec.EMA_DECAY == 0.998

    def test_without_out_the_policy_goes_to_standard_output(self, capsys):
        assert dev_recipe().main([*self.ARGS, "--name", "to_stdout", "--set", "LS_EPS=0.1"]) == 0
        assert _constants(capsys.readouterr().out)["LS_EPS"] == 0.1

    @pytest.mark.parametrize("arguments, message", [
        (["--name", "_hidden"], "name"),
        (["--name", "ok", "--set", "NO_SUCH=1"], "NO_SUCH"),
        (["--name", "ok", "--set", "EMA_DECAY"], "NAME=VALUE"),
        (["--name", "ok", "--set", "EMA_DECAY=0.9.9"], "EMA_DECAY"),
        (["--name", "ok", "--set", "EMA_DECAY=0.9", "--set", "EMA_DECAY=0.8"], "EMA_DECAY"),
    ])
    def test_a_refusal_is_printed_and_exits_non_zero(self, arguments, message, capsys):
        assert dev_recipe().main([*self.ARGS, *arguments]) != 0
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize("content", [
        None,  # no such file
        "x = '__NAME__ __BASE_COMMIT__ __NODE_ID__ __CREATED_AT__'\ndef broken(:\n",
    ])
    def test_a_template_that_cannot_be_read_or_parsed_is_refused_cleanly(self, tmp_path, capsys, content):
        template = tmp_path / "template.py"
        if content is not None:
            template.write_text(content)
        assert dev_recipe().main([*self.ARGS, "--name", "ok", "--template", str(template)]) == 1
        assert capsys.readouterr().err.startswith("dev_recipe: ")


# --- the policy smoke ----------------------------------------------------------------------

#: The recipe's ARM constant for each arm name the policy smoke knows.
SMOKE_ARMS = {"abmil": "abmil", "clam": "clam_mb", "dtfd": "dtfd_mil", "nnmil": "simple_mil"}
#: Per arm: everything off, everything the arm takes on together, and each of those alone.
SMOKE_CASES = [
    (arm, config, constants)
    for arm, recipe_arm in SMOKE_ARMS.items()
    for config, constants in {"all_off": {}, "all_on": all_on(recipe_arm), **switches_for(recipe_arm)}.items()
]


def _smoke(tmp_path, arm, constants, *extra):
    path = write_recipe(tmp_path, ARM=SMOKE_ARMS[arm], **constants)
    return policy_smoke.main(["--task-family", "classification", "--arm", arm, *extra, str(path)])


class TestPolicySmoke:
    @pytest.mark.parametrize(
        "arm, config, constants", SMOKE_CASES, ids=[f"{arm}-{config}" for arm, config, _ in SMOKE_CASES],
    )
    def test_the_smoke_passes_for_every_arm_and_switch(self, tmp_path, capsys, arm, config, constants):
        assert _smoke(tmp_path, arm, constants) == 0, capsys.readouterr().err

    def test_the_smoke_passes_without_naming_an_arm(self, tmp_path, capsys):
        path = write_recipe(tmp_path, **all_on("abmil"))  # the template's own ARM
        assert policy_smoke.main([str(path)]) == 0, capsys.readouterr().err

    def test_the_recipe_overrides_every_seam_the_trainers_call(self, tmp_path):
        policy = recipe_class(tmp_path)
        for seam in ("transform_bag", "train_step", "before_validation"):
            assert defines_seam(policy, seam), seam

    def test_survival_cells_refuse_the_recipe_because_their_trainers_call_none_of_its_seams(
        self, tmp_path, capsys,
    ):
        path = write_recipe(tmp_path, ARM="abmil")
        assert policy_smoke.main(["--task-family", "survival", "--arm", "abmil", str(path)]) == 1
        assert "not wired for survival" in capsys.readouterr().err

    @needs_cuda
    @pytest.mark.parametrize("arm", SMOKE_ARMS)
    def test_every_switch_on_passes_the_smoke_on_the_trainers_device(self, tmp_path, capsys, arm):
        assert policy_smoke._device().type == "cuda"
        assert _smoke(tmp_path, arm, all_on(SMOKE_ARMS[arm])) == 0, capsys.readouterr().err


# --- the real arms ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRun:
    """One real trainer on its tiny fixture: ``run(workdir, runtime)`` returns a comparable record."""

    name: str
    recipe_arm: str
    run: Callable[[Path, object], object]


def _nnmil(model_type: str) -> Callable[[Path, object], object]:
    return lambda workdir, runtime: nnmil_seams._train(workdir, runtime, model_type=model_type)[1]


RECIPE_ARM = {
    "abmil": "abmil", "dtfd": "dtfd_mil", "titan": "abmil",
    "clam": "clam_mb", "clam-without-instance-clustering": "clam_mb",
}
RUNS = (
    *(ArmRun(arm.name, RECIPE_ARM[arm.name], arm.run) for arm in ARMS),
    ArmRun("nnmil", "simple_mil", _nnmil("simple_mil")),
    ArmRun("nnmil-vision_transformer", "simple_mil", _nnmil("vision_transformer")),
)
BY_NAME = {run.name: run for run in RUNS}

_NATIVE: dict[str, tuple] = {}


def _native(run: ArmRun, factory):
    """The arm's native run (result and printed trajectory), computed once per arm."""
    if run.name not in _NATIVE:
        _NATIVE[run.name] = captured(run.run, factory.mktemp(f"native_{run.name}") / "run", fold_runtime())
    return _NATIVE[run.name]


def _recipe_run(run: ArmRun, tmp_path, **constants):
    """The arm's run under a recipe with ``constants``: the policy file lives in ``tmp_path``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = recipe_class(tmp_path, ARM=run.recipe_arm, **constants)
    return captured(run.run, tmp_path / "live", fold_runtime(policy))


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_every_switch_off_is_the_native_run(run, tmp_path, tmp_path_factory):
    assert _recipe_run(run, tmp_path) == _native(run, tmp_path_factory)


def _applies(*excluded: str) -> tuple[str, ...]:
    return tuple(name for name in BY_NAME if name not in excluded and name != "nnmil-vision_transformer")


#: Where each switch has something to act on in the arms' tiny fixtures. The tile caps
#: do nothing on a one-vector slide (TITAN) and are below nnMIL's 8-tile floor on its
#: bags of 3 to 6 (``test_the_tile_cap_reaches_nnmils_padded_batches`` lowers the floor);
#: UNFREEZE needs the widest layer to take the feature width, which the CLAM and DTFD
#: fixtures' 64-wide features do not give (their production shapes are checked in
#: ``test_recipe_units.py``); BODY_LR_SCALE is nnMIL's alone and needs a layer besides the
#: last one, which the stand-in of ``test_unfreeze_and_the_body_scale_change_the_weights_nnmil_learns``
#: has and the base fixtures do not.
APPLIES = {
    "ema": _applies(),
    "label_smoothing": _applies(),
    "positive_share": _applies(),
    "class_weights": _applies(),
    "warmup": _applies(),
    "cosine": _applies(),
    "step_anneal": _applies(),
    "max_tiles": _applies("titan", "nnmil"),
    "keep_frac": _applies("titan", "nnmil"),
    "unfreeze": ("abmil", "titan", "nnmil"),
    "body_lr_scale": (),
}
EFFECT_CASES = [(switch, name) for switch, names in APPLIES.items() for name in names]


@pytest.mark.parametrize("switch, name", EFFECT_CASES)
def test_a_switch_changes_the_run_it_applies_to(switch, name, tmp_path, tmp_path_factory):
    run = BY_NAME[name]
    live = _recipe_run(run, tmp_path, **SWITCHES[switch])
    assert live != _native(run, tmp_path_factory), f"{switch} is a silent no-op on {name}"


@pytest.mark.parametrize("constants", [{"MAX_TILES": 3, "MIN_TILES": 2}, {"KEEP_FRAC": 0.5, "MIN_TILES": 2}])
def test_the_tile_caps_are_refused_on_nnmil_before_a_batch_is_built(constants, tmp_path):
    """nnMIL re-pads every transformed bag and pools the zero rows: the recipe refuses the caps there."""
    with pytest.raises(ValueError, match="simple_mil"):
        _recipe_run(BY_NAME["nnmil"], tmp_path, **constants)


class _BodyHeadBag(torch.nn.Module):
    """nnMIL's mean-over-the-bag stand-in with a layer before the head, so there is a body to scale."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(nnmil_seams.WIDTH, 4)
        self.head = torch.nn.Linear(4, 2)
        self.batches: list = []
        self.masks: list = []

    def forward(self, features, **kwargs):
        if self.training:
            self.batches.append(features.detach().clone())
            if "mask" in kwargs:
                self.masks.append(kwargs["mask"].clone())
        return self.head(torch.relu(self.body(features.float().mean(dim=1))))


def test_unfreeze_and_the_body_scale_change_the_weights_nnmil_learns(tmp_path, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(nnmil_seams, "_MeanBag", _BodyHeadBag)
    run = ArmRun("nnmil-body-head", "simple_mil", _nnmil("simple_mil"))
    native, _ = captured(run.run, tmp_path_factory.mktemp("native_body_head") / "run", fold_runtime())
    assert native["weights"], "the stand-in has weights to compare"
    for constants in ({"UNFREEZE": (1, 3, 0.2)}, {"BODY_LR_SCALE": 0.2}):
        live, _ = _recipe_run(run, tmp_path, **constants)
        assert live["weights"] != native["weights"], constants


def test_a_body_scale_needs_a_body(tmp_path):
    """nnMIL's one-layer stand-in is all head."""
    with pytest.raises(ValueError, match="BODY_LR_SCALE found no body"):
        _recipe_run(BY_NAME["nnmil"], tmp_path, BODY_LR_SCALE=0.2)


@pytest.mark.parametrize("name", ["abmil", "dtfd", "clam", "titan"])
def test_a_body_scale_is_refused_on_every_arm_but_nnmil(name, tmp_path):
    with pytest.raises(ValueError, match="BODY_LR_SCALE.*simple_mil only"):
        _recipe_run(BY_NAME[name], tmp_path, BODY_LR_SCALE=0.2)


@pytest.mark.parametrize("name", ["clam", "clam-without-instance-clustering", "dtfd"])
def test_unfreeze_refuses_a_fixture_whose_widest_layer_is_not_the_feature_projection(name, tmp_path):
    with pytest.raises(ValueError, match="UNFREEZE"):
        _recipe_run(BY_NAME[name], tmp_path, **SWITCHES["unfreeze"])


def test_every_switch_on_together_trains_abmil_and_nnmil_to_completion(tmp_path, tmp_path_factory, monkeypatch):
    abmil = BY_NAME["abmil"]
    result, lines = _recipe_run(abmil, tmp_path / "abmil", **all_on("abmil"))
    assert len(lines) == len(_native(abmil, tmp_path_factory)[1]) and result != _native(abmil, tmp_path_factory)[0]

    monkeypatch.setattr(nnmil_seams, "_MeanBag", _BodyHeadBag)  # nnMIL needs a body for BODY_LR_SCALE
    record, lines = _recipe_run(
        ArmRun("nnmil", "simple_mil", _nnmil("simple_mil")), tmp_path / "nnmil", **all_on("simple_mil"),
    )
    assert len(lines) == nnmil_seams.EPOCHS and record["weights"]

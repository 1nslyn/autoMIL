"""H-1 (audit 2026-07-23): the raw training stdout captured to the agent-visible
archive/<node>/run.log must not carry sealed test metrics, and neither must the
log tail the daemon copies into the agent-facing result["error"].

The val-firewall previously relied on every training script (and every vendored
dependency) self-censoring its test prints — a re-vendored upstream would silently
reintroduce the leak.
"""
from __future__ import annotations

from automil.firewall import (
    REDACTION,
    held_out_keys,
    redact_held_out,
    redact_log_file,
)


def test_redacts_common_test_metric_prints():
    text = "\n".join([
        "epoch 1: val_auc=0.71 val_loss=0.60",
        "Test error: 0.2500, ROC AUC: 0.8123",
        "epoch 2: val_auc=0.73",
        "test_c_index=0.6421",
    ])
    out = redact_held_out(text)
    assert "0.8123" not in out
    assert "0.6421" not in out
    # Validation lines survive — this is a debugging surface, not a blackout.
    assert "val_auc=0.71" in out
    assert "val_auc=0.73" in out
    assert out.count(REDACTION) == 2


def test_redacts_consumer_specific_held_out_keys():
    result = {"held_out": {"test_kappa": 0.44}}
    keys = held_out_keys(result)
    assert keys == ("test_kappa",)
    out = redact_held_out("final test_kappa = 0.44", keys)
    assert "0.44" not in out


def test_clean_log_is_untouched():
    text = "epoch 1: val_auc=0.71\nepoch 2: val_auc=0.73\n"
    assert redact_held_out(text) == text.rstrip("\n")


def test_redact_log_file_rewrites_in_place(tmp_path):
    p = tmp_path / "run.log"
    p.write_text("val_auc=0.7\nTest error: 0.25, ROC AUC: 0.91\n")
    n = redact_log_file(p, ())
    assert n == 1
    body = p.read_text()
    assert "0.91" not in body
    assert "val_auc=0.7" in body


def test_redact_log_file_missing_is_noop(tmp_path):
    assert redact_log_file(tmp_path / "absent.log", ()) == 0


def test_empty_text_safe():
    assert redact_held_out("") == ""
    assert held_out_keys({}) == ()
    assert held_out_keys({"held_out": None}) == ()

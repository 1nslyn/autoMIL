"""Tests for the gated-attention ABMIL (faithful port of Ilse et al., 2018).

Covers the model directly (benchmarks/lib/nnMIL/.../models/ab_mil_gated.py)
and its registration under the `ab_mil` factory key
(benchmarks/lib/nnMIL/.../model_factory.py).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from autobench import LIB_ROOT  # noqa: E402

_LIB_DIR = str(LIB_ROOT)
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from nnMIL.network_architecture.model_factory import create_mil_model  # noqa: E402
from nnMIL.network_architecture.models.ab_mil_gated import AB_MIL_Gated  # noqa: E402


IN_DIM = 32
N_INSTANCES = 17
BATCH = 4


def _make_model(num_classes=2, in_dim=IN_DIM, M=16, L=8):
    """Small dims for fast tests; production dims (500/128) checked separately."""
    return AB_MIL_Gated(in_dim=in_dim, M=M, L=L, num_classes=num_classes, dropout=0.0, K=1)


def _random_bag(batch=BATCH, n=N_INSTANCES, in_dim=IN_DIM):
    return torch.randn(batch, n, in_dim, requires_grad=False)


class TestForwardShape:
    def test_forward_returns_logits_key(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x)
        assert isinstance(out, dict)
        assert "logits" in out

    def test_logits_shape_binary(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x)
        assert out["logits"].shape == (BATCH, 2)

    @pytest.mark.parametrize("num_classes", [2, 3, 6, 7])
    def test_logits_shape_multiclass(self, num_classes):
        """Must work for binary (num_classes=2) AND multi-class (>2, e.g. CLWD 6/7-class)."""
        model = _make_model(num_classes=num_classes)
        x = _random_bag()
        out = model(x)
        assert out["logits"].shape == (BATCH, num_classes)

    def test_no_sigmoid_probabilities_emitted(self):
        """Logits, not bounded [0,1] sigmoid probabilities (unlike the reference's binary head)."""
        model = _make_model(num_classes=2)
        x = _random_bag() * 10  # scale up so unclipped logits would likely exceed [0, 1]
        out = model(x)
        logits = out["logits"]
        assert (logits < 0).any() or (logits > 1).any()

    def test_variable_bag_size(self):
        model = _make_model(num_classes=2)
        for n in [1, 5, 50]:
            x = torch.randn(2, n, IN_DIM)
            out = model(x)
            assert out["logits"].shape == (2, 2)


class TestReturnDictContract:
    def test_return_WSI_feature(self):
        model = _make_model(num_classes=2, M=16)
        x = _random_bag()
        out = model(x, return_WSI_feature=True)
        assert "WSI_feature" in out
        assert out["WSI_feature"].shape == (BATCH, 16 * 1)  # K*M

    def test_return_WSI_attn(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True)
        assert "WSI_attn" in out
        assert out["WSI_attn"].shape == (BATCH, N_INSTANCES, 1)  # [B, N, K]

    def test_extra_keys_absent_by_default(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x)
        assert "WSI_feature" not in out
        assert "WSI_attn" not in out

    def test_both_extras_together(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True, return_WSI_feature=True)
        assert set(out.keys()) == {"logits", "WSI_feature", "WSI_attn"}


class TestAttentionWeights:
    def test_attention_non_negative_and_sums_to_one(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True)
        attn = out["WSI_attn"]  # [B, N, K]
        assert (attn >= 0).all()
        summed = attn.sum(dim=1)  # sum over N instances -> [B, K]
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)

    @pytest.mark.parametrize("n", [1, 3, 100])
    def test_attention_sums_to_one_various_bag_sizes(self, n):
        model = _make_model(num_classes=2)
        x = torch.randn(2, n, IN_DIM)
        out = model(x, return_WSI_attn=True)
        summed = out["WSI_attn"].sum(dim=1)
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


class TestGradientFlow:
    def test_gradients_reach_projection(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        proj_linear = model.feature_extractor[0]
        assert proj_linear.weight.grad is not None
        assert torch.any(proj_linear.weight.grad != 0)

    def test_gradients_reach_attention_V_branch(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        v_linear = model.attention_V[0]
        assert v_linear.weight.grad is not None
        assert torch.any(v_linear.weight.grad != 0)

    def test_gradients_reach_attention_U_branch(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        u_linear = model.attention_U[0]
        assert u_linear.weight.grad is not None
        assert torch.any(u_linear.weight.grad != 0)

    def test_gradients_reach_attention_w(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        assert model.attention_w.weight.grad is not None
        assert torch.any(model.attention_w.weight.grad != 0)

    def test_gradients_reach_classifier(self):
        model = _make_model(num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        assert model.classifier.weight.grad is not None
        assert torch.any(model.classifier.weight.grad != 0)

    def test_gradients_reach_all_params_via_ce_loss(self):
        """End-to-end: cross-entropy loss (the real training objective) reaches every param."""
        model = _make_model(num_classes=3)
        x = _random_bag()
        y = torch.randint(0, 3, (BATCH,))
        logits = model(x)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.any(p.grad != 0), f"gradient is all-zero for {name}"


class TestGatedAttentionFidelity:
    """Distinguish this from non-gated attention: BOTH a Tanh and a Sigmoid branch must exist."""

    def test_has_tanh_branch(self):
        model = _make_model(num_classes=2)
        assert isinstance(model.attention_V[1], torch.nn.Tanh)

    def test_has_sigmoid_branch(self):
        model = _make_model(num_classes=2)
        assert isinstance(model.attention_U[1], torch.nn.Sigmoid)

    def test_gating_is_multiplicative(self):
        """A = attention_w(V(H) * U(H)) — element-wise product of both branches, per the reference."""
        model = _make_model(num_classes=2, M=16, L=8)
        x = _random_bag()
        H = model.feature_extractor(x)
        A_V = model.attention_V(H)
        A_U = model.attention_U(H)
        expected_pre_softmax = model.attention_w(A_V * A_U).transpose(-1, -2)
        actual = torch.nn.functional.softmax(expected_pre_softmax, dim=-1)
        out = model(x, return_WSI_attn=True)
        actual_attn = out["WSI_attn"].transpose(-1, -2)
        assert torch.allclose(actual, actual_attn, atol=1e-6)

    def test_zeroing_either_gate_changes_attention(self):
        """Sanity check both branches actually participate: removing either changes A."""
        model = _make_model(num_classes=2, M=16, L=8)
        x = _random_bag()
        with torch.no_grad():
            baseline = model(x, return_WSI_attn=True)["WSI_attn"].clone()
            model.attention_U[0].weight.zero_()
            model.attention_U[0].bias.zero_()
            after_zero_U = model(x, return_WSI_attn=True)["WSI_attn"]
        assert not torch.allclose(baseline, after_zero_U, atol=1e-6)


class TestPaperExactDims:
    def test_default_factory_dims_are_paper_exact(self):
        """Locked decision: M=500, L=128 (matches Ilse et al. 2018), not 512/128."""
        model = create_mil_model(model_type="ab_mil", input_dim=1024, num_classes=2)
        assert isinstance(model, AB_MIL_Gated)
        assert model.M == 500
        assert model.L == 128

    def test_factory_forward_smoke(self):
        model = create_mil_model(model_type="ab_mil", input_dim=64, num_classes=4)
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert out["logits"].shape == (2, 4)


class TestInputValidation:
    def test_rejects_wrong_ndim(self):
        model = _make_model(num_classes=2)
        with pytest.raises(ValueError):
            model(torch.randn(N_INSTANCES, IN_DIM))  # missing batch dim

    def test_rejects_wrong_feature_dim(self):
        model = _make_model(num_classes=2, in_dim=IN_DIM)
        with pytest.raises(ValueError):
            model(torch.randn(BATCH, N_INSTANCES, IN_DIM + 1))

    def test_rejects_k_other_than_one(self):
        with pytest.raises(ValueError):
            AB_MIL_Gated(in_dim=IN_DIM, num_classes=2, K=2)

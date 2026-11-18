import torch

from ultralytics_ext.attention_kd import SpatialAttentionKDLoss
from ultralytics_ext.composite_kd import CompositeKDConfig, CompositeKDLoss
from ultralytics_ext.foreground_relation_kd import MaskedChannelRelationKDLoss
from ultralytics_ext.response_kd import ResponseLocalizationKDLoss


def features(requires_grad=False):
    return {15: torch.randn(2, 32, 40, 40, requires_grad=requires_grad), 18: torch.randn(2, 24, 20, 20, requires_grad=requires_grad), 21: torch.randn(2, 16, 10, 10, requires_grad=requires_grad)}


def batch():
    return {"bboxes": torch.tensor([[.5, .5, .4, .3], [.25, .25, .2, .2]]), "batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([[0.], [2.]])}


def test_attention_gradient_is_student_only():
    s, t = features(True), features(True); loss, _ = SpatialAttentionKDLoss()(s, t); loss.backward()
    assert all(x.grad is not None for x in s.values()); assert all(x.grad is None for x in t.values())


def test_soft_foreground_and_background_finite():
    for mode in ("foreground", "background"):
        s, t = features(True), features(); loss, _ = MaskedChannelRelationKDLoss(mode=mode)(s, t, batch())
        assert torch.isfinite(loss); loss.backward(); assert any(x.grad is not None for x in s.values())


def test_empty_foreground_is_zero_and_differentiable():
    s, t = features(True), features(); empty = {"bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0), "cls": torch.empty(0, 1)}
    loss, _ = MaskedChannelRelationKDLoss()(s, t, empty); assert loss == 0; loss.backward()


def test_response_shapes_and_gradient():
    s = {"scores": torch.randn(2, 3, 120, requires_grad=True), "boxes": torch.randn(2, 64, 120, requires_grad=True)}
    t = {"scores": torch.randn(2, 3, 120, requires_grad=True), "boxes": torch.randn(2, 64, 120, requires_grad=True)}
    loss, parts = ResponseLocalizationKDLoss()(s, t); loss.backward(); assert set(parts) == {"classification", "localization"}
    assert s["scores"].grad is not None and s["boxes"].grad is not None and t["scores"].grad is None


def test_composite_global_attention():
    cfg = CompositeKDConfig("unit", {"gram": .03, "attention": .01}); loss = CompositeKDLoss(cfg)
    raw, _ = loss.raw_components(features(True), features(), batch()); total = sum(raw.values()); total.backward(); assert torch.isfinite(total)

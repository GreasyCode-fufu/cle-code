"""Unit tests for the parameter-free channel-relation KD objective."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ultralytics_ext.relation_kd import ChannelRelationKDLoss


LAYERS = (15, 18, 21)
CHANNELS = (8, 12, 16)


def make_features(
    *,
    batch: int = 2,
    spatial: tuple[tuple[int, int], ...] = ((24, 24), (16, 16), (10, 10)),
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> dict[int, torch.Tensor]:
    return {
        index: torch.randn(batch, channels, height, width, dtype=dtype, requires_grad=requires_grad)
        for index, channels, (height, width) in zip(LAYERS, CHANNELS, spatial)
    }


def test_identical_features_have_zero_loss() -> None:
    features = make_features()
    loss, parts = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(features, features)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert all(value.item() == pytest.approx(0.0, abs=1e-8) for value in parts.values())


def test_different_features_have_positive_loss() -> None:
    student = make_features()
    teacher = make_features()
    loss, _ = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_backward_gives_only_student_finite_nonzero_gradients() -> None:
    student = make_features(requires_grad=True)
    teacher = make_features(requires_grad=True)
    loss, _ = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)
    loss.backward()
    assert all(
        tensor.grad is not None
        and torch.isfinite(tensor.grad).all()
        and torch.count_nonzero(tensor.grad).item() > 0
        for tensor in student.values()
    )
    assert all(tensor.grad is None for tensor in teacher.values())


def test_adaptive_pool_accepts_different_spatial_sizes_between_scales() -> None:
    spatial = ((37, 29), (19, 17), (9, 7))
    student = make_features(spatial=spatial)
    teacher = make_features(spatial=spatial)
    loss, parts = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)
    assert torch.isfinite(loss)
    assert set(parts) == {"p3", "p4", "p5"}


def test_corresponding_shape_mismatch_is_rejected() -> None:
    student = make_features()
    teacher = make_features()
    teacher[18] = torch.randn(2, CHANNELS[1] + 1, 16, 16)
    with pytest.raises(ValueError, match="shape mismatch"):
        ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_features_are_rejected(bad_value: float) -> None:
    student = make_features()
    teacher = make_features()
    student[15][0, 0, 0, 0] = bad_value
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)


def test_float16_inputs_are_computed_in_float32() -> None:
    student = make_features(dtype=torch.float16, requires_grad=True)
    teacher = make_features(dtype=torch.float16)
    loss, parts = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)
    assert loss.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in parts.values())
    loss.backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in student.values())


def test_relation_loss_has_no_trainable_parameters() -> None:
    loss_module = ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)
    assert sum(parameter.numel() for parameter in loss_module.parameters()) == 0
    assert not list(loss_module.named_parameters())


def test_bmm_constructs_channel_gram_not_spatial_gram() -> None:
    student = make_features(batch=2)
    teacher = make_features(batch=2)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_bmm = torch.bmm

    def recording_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original_bmm(left, right)

    with patch("torch.bmm", side_effect=recording_bmm):
        ChannelRelationKDLoss(layer_indices=LAYERS, pool_size=8)(student, teacher)

    assert len(calls) == 2 * len(LAYERS)
    for position, channels in enumerate(CHANNELS):
        for left, right in calls[2 * position : 2 * position + 2]:
            assert left == (2, channels, 64)
            assert right == (2, 64, channels)
            assert left[1] != 64 or channels == 64


"""Unit tests for Multi-step Transition Relation Distillation (MTRD)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ultralytics_ext.markov_relation_kd import MultiStepTransitionKDLoss


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


def loss_module(**overrides) -> MultiStepTransitionKDLoss:
    values = {"layer_indices": LAYERS, "pool_size": 8, "temperature": 0.2}
    values.update(overrides)
    return MultiStepTransitionKDLoss(**values)


def test_identical_features_have_zero_step_and_total_losses() -> None:
    features = make_features()
    loss, parts = loss_module()(features, features)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    for name in ("p3", "p4", "p5"):
        assert parts[f"{name}_step1"].item() == pytest.approx(0.0, abs=1e-7)
        assert parts[f"{name}_step2"].item() == pytest.approx(0.0, abs=1e-7)
        assert parts[f"{name}_total"].item() == pytest.approx(0.0, abs=1e-7)


def test_step_one_only_is_supported() -> None:
    features = make_features()
    loss, parts = loss_module(transition_steps=(1,), step_weights=(1.0,))(features, features)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert "p3_step1" in parts and "p3_step2" not in parts


def test_different_random_features_have_positive_loss() -> None:
    loss, _ = loss_module()(make_features(), make_features())
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_q_and_p_are_nonnegative_and_row_stochastic() -> None:
    feature = make_features()[15]
    q, p = loss_module().transition_matrices(feature)
    assert torch.all(q >= 0) and torch.all(p >= 0)
    assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-6)


def test_beta_zero_removes_q_and_p_diagonals() -> None:
    q, p = loss_module(self_loop_prob=0.0).transition_matrices(make_features()[15])
    assert torch.count_nonzero(q.diagonal(dim1=-2, dim2=-1)).item() == 0
    assert torch.count_nonzero(p.diagonal(dim1=-2, dim2=-1)).item() == 0


def test_fixed_self_loop_has_requested_diagonal() -> None:
    _, p = loss_module(self_loop_prob=0.25).transition_matrices(make_features()[15])
    assert torch.allclose(p.diagonal(dim1=-2, dim2=-1), torch.full_like(p[..., 0], 0.25))


def test_second_power_is_nonnegative_and_row_stochastic() -> None:
    _, p = loss_module().transition_matrices(make_features()[15])
    p2 = loss_module().transition_powers(p)[2]
    assert torch.all(p2 >= 0)
    assert torch.allclose(p2.sum(-1), torch.ones_like(p2.sum(-1)), atol=1e-6)


def test_student_has_finite_nonzero_gradient_and_teacher_has_none() -> None:
    student = make_features(requires_grad=True)
    teacher = make_features(requires_grad=True)
    loss, _ = loss_module()(student, teacher)
    loss.backward()
    assert all(
        value.grad is not None
        and torch.isfinite(value.grad).all()
        and torch.count_nonzero(value.grad).item() > 0
        for value in student.values()
    )
    assert all(value.grad is None for value in teacher.values())


def test_loss_and_parts_are_float32() -> None:
    loss, parts = loss_module()(make_features(), make_features())
    assert loss.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in parts.values())


def test_float16_inputs_are_stable_with_autocast_disabled_internally() -> None:
    student = make_features(dtype=torch.float16, requires_grad=True)
    teacher = make_features(dtype=torch.float16)
    loss, _ = loss_module()(student, teacher)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in student.values())


def test_different_spatial_sizes_across_scales_pool_normally() -> None:
    spatial = ((37, 29), (19, 17), (9, 7))
    loss, _ = loss_module()(make_features(spatial=spatial), make_features(spatial=spatial))
    assert torch.isfinite(loss)


def test_corresponding_shape_mismatch_is_rejected_immediately() -> None:
    student = make_features()
    teacher = make_features()
    teacher[18] = torch.randn(2, CHANNELS[1] + 1, 16, 16)
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_module()(student, teacher)


def test_fewer_than_two_channels_is_rejected() -> None:
    module = MultiStepTransitionKDLoss(layer_indices=(15,), layer_weights=(1.0,), pool_size=8)
    with pytest.raises(ValueError, match="at least 2 channels"):
        module.transition_matrices(torch.randn(2, 1, 8, 8))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_input_is_rejected(bad_value: float) -> None:
    features = make_features()
    features[15][0, 0, 0, 0] = bad_value
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        loss_module()(features, make_features())


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_nonpositive_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        loss_module(temperature=temperature)


@pytest.mark.parametrize("steps", [(), (0,), (2,), (1, 1), (2, 1)])
def test_invalid_transition_steps_are_rejected(steps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="transition_steps"):
        loss_module(transition_steps=steps, step_weights=(1.0,) * len(steps))


def test_step_weights_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="step_weights length"):
        loss_module(transition_steps=(1, 2), step_weights=(1.0,))


@pytest.mark.parametrize("probability", [-0.01, 1.0, 1.2])
def test_invalid_self_loop_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        loss_module(self_loop_prob=probability)


def test_loss_has_no_trainable_parameters() -> None:
    module = loss_module()
    assert not list(module.named_parameters())
    assert sum(parameter.numel() for parameter in module.parameters()) == 0


def test_only_channel_relation_matrices_are_constructed() -> None:
    student = make_features(batch=2)
    teacher = make_features(batch=2)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_bmm = torch.bmm

    def recording_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original_bmm(left, right)

    with patch("torch.bmm", side_effect=recording_bmm):
        loss_module()(student, teacher)

    assert calls
    spatial_elements = 8 * 8
    for left, right in calls:
        assert left[0] == 2 and right[0] == 2
        assert not (left[-2:] == (spatial_elements, spatial_elements))
        assert not (right[-2:] == (spatial_elements, spatial_elements))


def test_structured_metrics_include_requested_fields() -> None:
    _, parts = loss_module()(make_features(), make_features())
    expected = {
        f"{name}_{suffix}"
        for name in ("p3", "p4", "p5")
        for suffix in ("step1", "step2", "total")
    }
    expected |= {
        f"transition_{stat}_{name}"
        for name in ("p3", "p4", "p5")
        for stat in ("entropy", "max_prob")
    }
    assert expected <= set(parts)


def test_transition_diagnostics_are_finite_and_bounded() -> None:
    diagnostics = loss_module().transition_diagnostics(make_features()[15])
    for value in diagnostics.values():
        assert torch.isfinite(value).all()
    assert torch.all((diagnostics["normalized_entropy_rows"] >= 0) & (diagnostics["normalized_entropy_rows"] <= 1.0001))
    assert torch.all((diagnostics["max_probability_rows"] >= 0) & (diagnostics["max_probability_rows"] <= 1))
    assert torch.all((diagnostics["top5_mass_rows"] >= 0) & (diagnostics["top5_mass_rows"] <= 1.0001))

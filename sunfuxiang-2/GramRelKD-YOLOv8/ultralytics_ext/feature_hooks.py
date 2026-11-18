"""Safe, batch-scoped feature capture for Ultralytics detection models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ultralytics.utils.torch_utils import unwrap_model


class FeatureHookBank:
    """Capture the last Tensor emitted by selected ``DetectionModel.model`` layers."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: tuple[int, ...] = (15, 18, 21),
        detach: bool = False,
        name: str = "student",
    ) -> None:
        self.name = str(name)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must contain unique layer indices")
        self.detach = bool(detach)
        self._features: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {i: 0 for i in self.layer_indices}
        self._handles: list[Any] = []
        self._active = False
        self._removed = False

        base_model = unwrap_model(model)
        container = getattr(base_model, "model", None)
        if not isinstance(container, nn.Sequential):
            raise TypeError(
                f"{self.name}: expected unwrap_model(model).model to be nn.Sequential, "
                f"got {type(container).__name__}"
            )
        self.container = container
        for index in self.layer_indices:
            if index < 0 or index >= len(container):
                raise IndexError(f"{self.name}: layer index {index} is outside [0, {len(container) - 1}]")
            self._handles.append(container[index].register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active:
                return
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{self.name}: layer {index} hook expected Tensor, got {type(output).__name__}"
                )
            self._counts[index] += 1
            self._features[index] = output.detach() if self.detach else output

        return capture

    @property
    def active(self) -> bool:
        return self._active

    @property
    def removed(self) -> bool:
        return self._removed

    @property
    def counts(self) -> Mapping[int, int]:
        return dict(self._counts)

    def enable(self) -> None:
        if self._removed:
            raise RuntimeError(f"{self.name}: hooks have already been removed")
        self._active = True

    def disable(self) -> None:
        self._active = False

    def clear(self) -> None:
        """Release all feature references and reset per-batch trigger counts."""
        self._features.clear()
        self._counts = {i: 0 for i in self.layer_indices}

    def get(self) -> dict[int, torch.Tensor]:
        """Return current batch features, requiring every requested layer to exist."""
        missing = [i for i in self.layer_indices if i not in self._features]
        if missing:
            raise RuntimeError(f"{self.name}: missing hook outputs for layers {missing}")
        return {i: self._features[i] for i in self.layer_indices}

    def validate_once(self) -> None:
        """Require exactly one hook trigger at every requested layer."""
        bad = {i: self._counts[i] for i in self.layer_indices if self._counts[i] != 1}
        if bad:
            raise RuntimeError(f"{self.name}: expected one hook trigger per layer, got {bad}")
        self.get()

    def remove(self) -> None:
        """Remove hook handles and release any retained computation graph."""
        if self._removed:
            return
        self.disable()
        self.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._removed = True

    def __enter__(self) -> "FeatureHookBank":
        self.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()

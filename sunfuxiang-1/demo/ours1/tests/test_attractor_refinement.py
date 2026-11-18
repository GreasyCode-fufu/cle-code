import torch

from ultralytics import YOLO
from ultralytics.nn.modules import ObjectAttractorRefinement
from ultralytics.nn.tasks import DetectionModel


def test_attractor_refinement_shape():
    module = ObjectAttractorRefinement(nc=3, enabled=True, topk=5, num_steps=2)
    preds = torch.rand(2, 7, 20)
    out = module(preds)
    assert out.shape == (2, 7, 5)


def test_attractor_disabled_is_noop():
    module = ObjectAttractorRefinement(nc=3, enabled=False, topk=5, num_steps=2)
    preds = torch.rand(2, 7, 20)
    out = module(preds)
    assert out is preds


def test_attractor_yaml_build_and_dummy_forward():
    model = YOLO("ultralytics/cfg/models/v8/yolov8n-attractor.yaml")
    model.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    preds = y[0] if isinstance(y, tuple) else y
    assert preds.shape[0] == 1
    assert preds.shape[1] == 4 + model.model.model[-1].nc
    assert preds.shape[2] <= model.model.model[-1].attractor.topk


def test_standard_model_has_no_active_attractor():
    model = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False).eval()
    assert getattr(model.model[-1], "attractor", None) is None

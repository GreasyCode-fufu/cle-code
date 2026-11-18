# Round12: fixed CNN/ViT complementarity

Zero new neural trainings;45 reused source models,45 inner-only temperatures,6 fixed pipelines. No fixed val/test or relabeling.
Mature ensemble/calibration diagnostic, not new architecture or budget matching.

| Pipeline | Mean pooled OOF F1 | Malignant recall | NLL | Brier |
|---|---:|---:|---:|---:|
| resnet_bn_train | 0.859045 | 0.695035 | 0.364364 | 0.184765 |
| resnet_bn_frozen | 0.869416 | 0.695035 | 0.321734 | 0.170632 |
| dino_full | 0.873909 | 0.736407 | 0.326841 | 0.170946 |
| mix_train_dino | 0.885805 | 0.743499 | 0.284408 | 0.153466 |
| mix_frozen_dino | 0.883648 | 0.729314 | 0.272266 | 0.149395 |
| mix_resnets | 0.878551 | 0.710993 | 0.289859 | 0.157833 |

Complete preregistered primary follow-up gate: False.
Intervals condition on repeated-development cross-fitted sources; not external confirmation or clinical confidence.
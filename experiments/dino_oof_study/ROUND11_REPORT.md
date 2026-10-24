# Round11: patient OOF check of full DINO adaptation

15 new full-DINO neural models,30 reused CNNs,3 frozen-DINO OOF reference files. Original train only; fixed val/test excluded.
Cross-fitted internal system comparisons; not a new architecture or external confirmation.

| Pipeline | Seed | Pooled OOF F1 | Malignant recall | Patient mean accuracy |
|---|---:|---:|---:|---:|
| dino_full | 42 | 0.859762 | 0.705674 | 0.873629 |
| dino_full | 43 | 0.876629 | 0.728723 | 0.901632 |
| dino_full | 44 | 0.885335 | 0.774823 | 0.899065 |
| resnet_bn_train | 42 | 0.854736 | 0.721631 | 0.867370 |
| resnet_bn_train | 43 | 0.864475 | 0.695035 | 0.888839 |
| resnet_bn_train | 44 | 0.857924 | 0.668440 | 0.887497 |
| resnet_bn_frozen | 42 | 0.862836 | 0.671986 | 0.884874 |
| resnet_bn_frozen | 43 | 0.882800 | 0.739362 | 0.899860 |
| resnet_bn_frozen | 44 | 0.862610 | 0.673759 | 0.887572 |
| dino_frozen_ridge | 42 | 0.870690 | 0.732270 | 0.882288 |
| dino_frozen_ridge | 43 | 0.875017 | 0.735816 | 0.891358 |
| dino_frozen_ridge | 44 | 0.873465 | 0.735816 | 0.886601 |

Complete preregistered internal follow-up gate: False.
Conditional intervals do not cover all overlapping-training/selection/pretraining confounds.
# Round13: new-split secondary fusion replication

45 NEW source models in fixed seeds45/46/47,45 inner-only temperatures,6 unchanged pipelines. Same185 internal patients, no fixed val/test or relabeling.
Mature ensemble/calibration diagnostic, not new architecture or budget matching.

| Pipeline | Mean pooled OOF F1 | Malignant recall | NLL | Brier |
|---|---:|---:|---:|---:|
| resnet_bn_train | 0.857488 | 0.659574 | 0.341650 | 0.184766 |
| resnet_bn_frozen | 0.868730 | 0.713948 | 0.339069 | 0.174395 |
| dino_full | 0.877763 | 0.743499 | 0.305408 | 0.163660 |
| mix_train_dino | 0.884017 | 0.732861 | 0.275467 | 0.152449 |
| mix_frozen_dino | 0.884790 | 0.745863 | 0.271654 | 0.149407 |
| mix_resnets | 0.873827 | 0.697400 | 0.292244 | 0.161885 |

Complete preregistered primary follow-up gate: False.
Intervals condition on repeated-development cross-fitted sources; not external confirmation or clinical confidence.
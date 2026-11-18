# CLE round 4: original-pixel global/local views

Exploratory development-selected scores; intervals condition on fitted models, without model-selection/multiplicity correction. No test evaluation.

Primary reference is the newly trained global224 control with matched 512/448 preprocessing. Legacy baseline comparisons are contextual only.

| Variant | Mean val F1 | Malignant F1 | Delta vs matched global | Patient interval | Extra parameters |
|---|---:|---:|---:|---|---:|
| global224 | 0.883140 | 0.791976 | +0.000000 | [0.0, 0.0] | 0 |
| single448 | 0.880061 | 0.785453 | -0.003079 | [-0.021954621013387063, 0.01688430883900622] | 0 |
| tile_fixed | 0.885287 | 0.794593 | +0.002147 | [-0.014558148312272087, 0.019744066515082214] | 17024 |
| tile_attention | 0.880528 | 0.785334 | -0.002612 | [-0.023077485265531474, 0.016602961478993667] | 148736 |
| tile_context | 0.880436 | 0.789253 | -0.002704 | [-0.015341896891671116, 0.00899941639189824] | 148672 |
| tile_no_context | 0.883943 | 0.795533 | +0.000803 | [-0.011890077720139273, 0.011593474007430679] | 148672 |
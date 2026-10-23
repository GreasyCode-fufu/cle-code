# CLE round 3: shared-weight scale routing

Exploratory development-selected scores; intervals condition on fitted models, without model-selection/multiplicity correction. No test evaluation.

| Variant | Mean val F1 | Malignant F1 | Delta vs GAP | Patient interval | Extra parameters |
|---|---:|---:|---:|---|---:|
| dilated_control | 0.879564 | 0.781253 | -0.002393 | [-0.022691130437262955, 0.01950006561319154] | 0 |
| fixed_dual | 0.877324 | 0.775486 | -0.004634 | [-0.028694320191957005, 0.017938423311768512] | 7168 |
| sk_dual | 0.881109 | 0.787239 | -0.000849 | [-0.014171839640536008, 0.01337659035721689] | 106560 |
| cle_contrast | 0.879506 | 0.785522 | -0.002451 | [-0.02423310156110377, 0.015012427206997558] | 106560 |
| cle_no_dilation | 0.877971 | 0.783465 | -0.003987 | [-0.021507894328456643, 0.011671150845162364] | 106560 |
| static_dual | 0.880217 | 0.784604 | -0.001741 | [-0.023850072039683944, 0.019470063374904804] | 8192 |
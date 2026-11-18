# Method comparison

All new methods are exploratory extensions. The fixed official main method is **Gram RelKD**.

| Method | Split | mAP50-95 | mAP75 | Delta vs Gram | Assessment |
|---|---:|---:|---:|---:|---|
| Paired NoKD | test | 0.421200 | 0.435230 | -0.014480 | Reference |
| MTRD | test | 0.432186 | — | -0.003494 | Reference |
| Gram RelKD | test | 0.435680 | 0.459626 | +0.000000 | Official main method |
| E06_GramResponseKD_70_30 | test | 0.429919 | 0.460597 | -0.005761 | B — Complementary |
| E04_GramAdvP5_80_20 | test | 0.420200 | 0.437710 | -0.015480 | D — Ineffective |

E06 is complementary: crack and moisture AP50-95 improve over Gram RelKD, but overall mAP50-95 is lower and it does not replace Gram.
E04 is below Paired NoKD and its discriminator was one-sided, so adversarial supervision showed no added value in this run.

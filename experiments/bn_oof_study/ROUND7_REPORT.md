# Round 7: internal patient OOF normalization check

15 new frozen-BN trainings, 15 reused standard-BN models. No grades, fixed val or test used. Internal check, not external confirmation.

| Arm | Seed | Pooled OOF macro-F1 | Malignant recall | Patient mean accuracy |
|---|---:|---:|---:|---:|
| bn_train | 42 | 0.854736 | 0.721631 | 0.867370 |
| bn_train | 43 | 0.864475 | 0.695035 | 0.888839 |
| bn_train | 44 | 0.857924 | 0.668440 | 0.887497 |
| bn_frozen | 42 | 0.862836 | 0.671986 | 0.884874 |
| bn_frozen | 43 | 0.882800 | 0.739362 | 0.899860 |
| bn_frozen | 44 | 0.862610 | 0.673759 | 0.887572 |

{
  "seed_deltas": [
    0.008100004652686787,
    0.018325004393418998,
    0.004686579459580065
  ],
  "mean_delta": 0.010370529501895284,
  "patient_cluster95": [
    -0.005300387210399598,
    0.0262967095498221
  ],
  "patient_groups": 185,
  "resamples": 20000,
  "strong_diagnostic_signal": false
}

Intervals condition on cross-fitted models; overlapping fold-training sets, development selection and training uncertainty limit interpretation.
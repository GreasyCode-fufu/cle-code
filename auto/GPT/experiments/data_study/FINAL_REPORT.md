# Frozen CLE data study results

Completed: 2026-10-24T04:13:41.204702+08:00

All methods use the same ResNet34, three seeds, 60 epochs, preprocessing and complete test set.
A-E grades measure OOF agreement/learnability, not verified label correctness.

| Arm | Mean macro-F1 | Seed SD | Mean malignant F1 |
|---|---:|---:|---:|
| full | 0.872235 | 0.007277 | 0.827791 |
| ac_only | 0.898378 | 0.006821 | 0.863938 |
| grade_weighted | 0.883729 | 0.006166 | 0.841762 |

Paired comparisons (fraction units, multiply by 100 for percentage points):

```json
{
  "ac_only": {
    "seed_deltas": [
      0.017268499331028142,
      0.039316668932715904,
      0.02184424960452369
    ],
    "mean_delta": 0.02614313928942258,
    "patient_cluster95": [
      0.012097325742503606,
      0.04495483047701226
    ],
    "caveat": "Conditional on these three trained pairs; excludes grading and training-data resampling uncertainty."
  },
  "grade_weighted": {
    "seed_deltas": [
      0.01662994580138477,
      0.018942944334840672,
      -0.0010894426926488787
    ],
    "mean_delta": 0.011494482481192189,
    "patient_cluster95": [
      -0.010573497938319128,
      0.03366699231338596
    ],
    "caveat": "Conditional on these three trained pairs; excludes grading and training-data resampling uncertainty."
  }
}
```

The existing test set has previously been used in historical model selection. These are controlled comparisons, not a new untouched external validation.
No labels were changed; no validation/test images were removed; no method or threshold was selected using these test results.
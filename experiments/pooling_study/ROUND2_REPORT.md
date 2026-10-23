# CLE round 2: bounded spatial evidence pooling

Development-selected exploratory results only. Patient intervals condition on fitted pairs; selection/multiple comparisons need independent confirmation.
No test images were evaluated.

| Variant | Mean val F1 | Mean malignant F1 | Delta vs GAP | Patient interval | Extra parameters |
|---|---:|---:|---:|---|---:|
| gem_control | 0.880737 | 0.780495 | -0.001221 | [-0.02398034612778277, 0.02434126907182293] | 1 |
| attention_control | 0.879663 | 0.787036 | -0.002294 | [-0.019110988207173798, 0.013645311864611001] | 181136 |
| cle_evidence | 0.878639 | 0.781004 | -0.003319 | [-0.0241302151054686, 0.01636671119324418] | 180993 |
| cle_no_frequency | 0.882177 | 0.791781 | +0.000220 | [-0.016185729524051142, 0.017368080724468146] | 180992 |
| cle_fixed_blend | 0.873914 | 0.774661 | -0.008043 | [-0.026399075496266897, 0.008650154760859428] | 114881 |
| cle_no_context | 0.881569 | 0.790237 | -0.000389 | [-0.017266111639726314, 0.015699557862787934] | 148161 |
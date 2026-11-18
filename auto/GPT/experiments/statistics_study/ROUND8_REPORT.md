# Round 8 fixed-feature probes

Zero new CNN trainings; 30 reused encoders, 90 analytic ridge fits. No grades/fixed val/test.

| Encoder | Descriptor | Mean OOF F1 |
|---|---|---:|
| bn_train | gap | 0.862553 |
| bn_train | gap_std | 0.861927 |
| bn_train | gap_cov | 0.859587 |
| bn_frozen | gap | 0.871950 |
| bn_frozen | gap_std | 0.872489 |
| bn_frozen | gap_cov | 0.872603 |

{
  "bn_train_gap_std_vs_gap": {
    "seed_deltas": [
      -0.0036591524340182646,
      -0.003024578226539454,
      0.004805489977440569
    ],
    "mean_delta": -0.0006260802277057165,
    "patient95": [
      -0.003994743353837576,
      0.0025101822288512835
    ],
    "family_interval": [
      -0.004974250225790735,
      0.00341923319240407
    ],
    "strong_within_stratum": false
  },
  "bn_train_gap_cov_vs_gap": {
    "seed_deltas": [
      -0.005854343023074304,
      -0.0008498160750116668,
      -0.002194897996802947
    ],
    "mean_delta": -0.002966352364962973,
    "patient95": [
      -0.006552324122052893,
      0.0005442617187883767
    ],
    "family_interval": [
      -0.007557170324370453,
      0.0014681549032243006
    ],
    "strong_within_stratum": false
  },
  "bn_frozen_gap_std_vs_gap": {
    "seed_deltas": [
      -0.002771684692547627,
      0.00021428634267195434,
      0.004173577540072637
    ],
    "mean_delta": 0.0005387263967323216,
    "patient95": [
      -0.0026245315090390057,
      0.0035020887739102553
    ],
    "family_interval": [
      -0.003345629268715182,
      0.004294754389450288
    ],
    "strong_within_stratum": false
  },
  "bn_frozen_gap_cov_vs_gap": {
    "seed_deltas": [
      -0.0018913296483404896,
      0.0019508129681131692,
      0.001897486312529395
    ],
    "mean_delta": 0.0006523232107673582,
    "patient95": [
      -0.0028920500712762798,
      0.004508557056721937
    ],
    "family_interval": [
      -0.0037799918055881586,
      0.005585968884194131
    ],
    "strong_within_stratum": false
  }
}

Conditional internal diagnostic intervals; descriptor dimensions differ and scores are uncalibrated.
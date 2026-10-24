# Round14: strictly nested grading identity controls

225 linear grader fits; up-to60 new CNN students and15 reused baselines. Only original-train patients. No outer filtering.

| Pipeline | Seed | Pooled OOF F1 | Malignant recall | Patient mean accuracy |
|---|---:|---:|---:|---:|
| full_ce | 42 | 0.854736 | 0.721631 | 0.867370 |
| full_ce | 43 | 0.864475 | 0.695035 | 0.888839 |
| full_ce | 44 | 0.857924 | 0.668440 | 0.887497 |
| grade_weighted | 42 | 0.868595 | 0.698582 | 0.894587 |
| grade_weighted | 43 | 0.864359 | 0.684397 | 0.885115 |
| grade_weighted | 44 | 0.859672 | 0.686170 | 0.887546 |
| shuffled_weighted | 42 | 0.855933 | 0.668440 | 0.877769 |
| shuffled_weighted | 43 | 0.850958 | 0.640071 | 0.879627 |
| shuffled_weighted | 44 | 0.867297 | 0.689716 | 0.901113 |
| ac_only | 42 | 0.876550 | 0.693262 | 0.912746 |
| ac_only | 43 | 0.875710 | 0.716312 | 0.894155 |
| ac_only | 44 | 0.868427 | 0.707447 | 0.903553 |
| random_ac | 42 | 0.854785 | 0.640071 | 0.890602 |
| random_ac | 43 | 0.859241 | 0.675532 | 0.880202 |
| random_ac | 44 | 0.854749 | 0.654255 | 0.884003 |

{
  "comparisons": [
    {
      "candidate": "grade_weighted",
      "reference": "full_ce",
      "seed_deltas": [
        0.013859260676753471,
        -0.00011615237321283001,
        0.0017481956646239194
      ],
      "mean_delta": 0.0051637679893881865,
      "patient_cluster95": [
        -0.0062933277721893876,
        0.017170662109787023
      ],
      "family_interval": [
        -0.009480571324872802,
        0.02023437064568133
      ],
      "strong": false
    },
    {
      "candidate": "grade_weighted",
      "reference": "shuffled_weighted",
      "seed_deltas": [
        0.012662185552852145,
        0.013400953972482088,
        -0.007624749385957363
      ],
      "mean_delta": 0.006146130046458957,
      "patient_cluster95": [
        -0.0005760308689288119,
        0.01283546598508331
      ],
      "family_interval": [
        -0.002578214290355282,
        0.014860260288428686
      ],
      "strong": false
    },
    {
      "candidate": "ac_only",
      "reference": "full_ce",
      "seed_deltas": [
        0.021813379603335625,
        0.011235018605776137,
        0.010503576034178952
      ],
      "mean_delta": 0.014517324747763571,
      "patient_cluster95": [
        0.0026500978514777005,
        0.02736524821058981
      ],
      "family_interval": [
        -0.0002581726037453933,
        0.03101150978665984
      ],
      "strong": false
    },
    {
      "candidate": "ac_only",
      "reference": "random_ac",
      "seed_deltas": [
        0.021765020588230932,
        0.01646873816685479,
        0.013678923995162129
      ],
      "mean_delta": 0.01730422758341595,
      "patient_cluster95": [
        0.00827808948371062,
        0.027207776873758995
      ],
      "family_interval": [
        0.005882832395087748,
        0.029905005932678936
      ],
      "strong": true
    }
  ],
  "strong_primary_followup": false,
  "strong_secondary_followup": false,
  "resamples": 20000,
  "groups": 185
}

Grades are consistency proxies, not label-error truth. Internal conditional intervals do not cover all repeated development or overlapping-training uncertainty.
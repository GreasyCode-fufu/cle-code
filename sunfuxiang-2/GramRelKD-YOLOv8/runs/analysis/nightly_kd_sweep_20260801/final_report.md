# Nightly KD sweep report

## Outcome

Status: **PARTIAL_SUCCESS**. Six complete 300-epoch experiments finished in 7.558 hours with zero failed experiments and zero fallbacks.
E07_GramAdvP4P5_80_20 was not started because the runner's remaining-time estimate was below the required experiment time plus the 10-minute safety margin.

The fixed final main method remains **Gram RelKD**. All six new methods are exploratory extensions.

## Validation ranking

| Rank | Experiment | Val mAP50-95 | Val mAP75 | Moisture AP50-95 | Crack AP50-95 |
|---:|---|---:|---:|---:|---:|
| 1 | E06_GramResponseKD_70_30 | 0.434847 | 0.444849 | 0.326838 | 0.407039 |
| 2 | E04_GramAdvP5_80_20 | 0.431527 | 0.460944 | 0.312909 | 0.409288 |
| 3 | E02_GramSoftFG_70_30 | 0.428234 | 0.437579 | 0.300375 | 0.410171 |
| 4 | E05_GramDecoupledCB_60_25_15 | 0.428120 | 0.448109 | 0.315989 | 0.406819 |
| 5 | E03_GramAT_80_20 | 0.422504 | 0.429083 | 0.301897 | 0.404406 |
| 6 | E01_GramMTRD_75_25 | 0.422408 | 0.441988 | 0.303103 | 0.407866 |

E06 Gram+ResponseKD led validation at 0.434847, still slightly below the fixed Gram RelKD validation reference of 0.43553.

## Test selection and results

Test evaluation was restricted to the two validation leaders. E06 ranked first; E04 ranked second and was also the best/only completed GAN. No test result was used to retune a loss or launch another experiment.

| Experiment | P | R | mAP50 | mAP75 | mAP50-95 | Spalling | Crack | Moisture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E06_GramResponseKD_70_30 | 0.783187 | 0.664733 | 0.698775 | 0.460597 | 0.429919 | 0.544858 | 0.420706 | 0.324193 |
| E04_GramAdvP5_80_20 | 0.758543 | 0.663667 | 0.685703 | 0.437710 | 0.420200 | 0.532906 | 0.420047 | 0.307648 |

E06 is the best exploratory test model at mAP50-95=0.429919, versus Paired NoKD=0.421200, MTRD=0.432186, and Gram RelKD=0.435680.
Against Gram, E06 changes mAP50-95 by -0.005761, mAP75 by +0.000971, crack AP50-95 by +0.010025, and moisture AP50-95 by +0.009836.
It is therefore a complementary exploratory extension, not a replacement: weak classes improved, but overall accuracy and spalling declined.

## GAN assessment

E04 completed without NaN/Inf, gradient explosion, memory leakage, or checkpoint contamination, but its discriminator stayed one-sided (real accuracy near 1, fake accuracy near 0). Its test mAP50-95=0.420200 is below Paired NoKD and Gram RelKD. The GAN added no demonstrated value here.

## Class-specific findings

- Best exploratory moisture result among tested models: E06, AP50-95=0.324193 (Gram reference 0.314357).
- Best exploratory crack result among tested models: E06, AP50-95=0.420706 (Gram reference 0.410681).
- Best spalling result remains Gram RelKD, AP50-95=0.582001; E06 reached 0.544858.

## Integrity and cost

All experiments used the same initialization SHA256, fixed seed/data/training call chain, and unchanged Student inference graph. Fused/unfused parameters remained 2,603,129/2,607,913 and GFLOPs remained 7.044096.
Teacher, Ultralytics core, and all protected prior-result hashes are unchanged. All twelve best/last checkpoints passed key/module isolation checks; selected models also loaded and inferred without `ultralytics_ext`.
Training-only overhead is tabulated in full_results.csv. E05 was the most expensive (about 120 ms KD compute per batch); no inference overhead was introduced.

## Reporting correction

The automatic runner originally added 1 to an already 1-based `results.csv` epoch. This reporting-only error was corrected in sweep_state.json and full_results.csv; no metrics, weights, checkpoints, or training code were changed.

## Final decision

**Gram RelKD remains the exact final main method.** E06 is the only extension worth considering for a future explicitly approved multi-seed study, because it trades lower overall accuracy for better crack/moisture performance. No additional experiment was started.

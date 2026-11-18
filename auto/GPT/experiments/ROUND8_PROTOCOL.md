# Round 8: fixed-feature texture-statistics diagnosis

Preregistered 2026-10-24 before extracting diagnostic features or fitting new heads. No new CNN training, normalization variants or test evaluation.

## Motivation

Four architecture rounds did not confirm new effects; loss/transfer controls also did not meet gates. Patient OOF frozen BN showed +1.04pt in mean with all three seed directions positive but interval [-.53,+2.63]pt. Do not treat that unresolved signal as a validated recipe. Use both standard-BN and frozen-BN encoders as separate strata.

Earlier extreme-region/high-resolution alternatives did not establish benefit. Hypothesis: global means may omit useful heterogeneity/coactivation statistics even when selecting extreme regions is unhelpful. A fixed-encoder probe will test this before spending on a trainable covariance architecture. Positive probe results would support follow-up, not novelty or clinical morphology interpretation. Negative results do not exclude end-to-end learned statistics.

## Design

Reuse all 30 audited patient-OOF checkpoints from round 7: two encoders x three seeds x five folds. There are zero newly trained CNNs. Each checkpoint is frozen in eval; it extracts per-image layer4 features on the original center224 / resize256 input for original train2386 only. All probabilities at outer indices must reproduce the source CNN OOF probabilities. No fixed val/test image is loaded. The shared label-independent cache contains all original splits, but effective indices are 0..2385 only.

For each encoder/seed/fold compare three linear-head schemes, six schemes total (2 encoders x 3 descriptors), each three seeds / five folds. There are 90 analytic ridge fits, not 90 neural-network trainings:

1. gap: 512-dimensional layer4 spatial mean, the same-dimensional linear-head control.
2. gap_std: mean plus per-channel spatial standard deviation, 1024 dimensions.
3. gap_cov: mean plus centered covariance of a fixed orthogonal 512->32 random projection. Use upper triangle with off-diagonal sqrt(2) weighting and elementwise signed square root (not matrix square root), 512+528=1040 dimensions.

Projection generated once from label-blind numpy RNG seed20261024 and QR; freeze its array hash. Center within each image, unbiased denominator spatial_n-1. Standard deviation sqrt(max(var,0)+1e-6). Covariance signed power is sign(x)*sqrt(abs(x)+1e-12). No learned projection, matrix square root, ROI, attention or disease-specific cue is claimed. These are generic texture statistics inspired by established second-order representations.

Each descriptor block is centered/scaled using only this checkpoint's fit indices. Std floor1e-6, divide each block by sqrt(block_dim * number_of_blocks), so adding a block does not simply double its expected squared norm. No outer/inner data enters mean/std estimation. Fit multi-output Ridge(alpha=1, fit_intercept=True, solver=cholesky), targets original one-hot classes, no class/sample weights and no hyperparameter search. The original backbone checkpoint was selected by inner patients; the probe uses only fit patients, with inner patients unused for head fitting or tuning. Outer patients are predicted once after fitting. Record softmax of ridge scores as normalized scores, explicitly not calibrated probabilities.

No neural representation is updated. Different descriptor dimensions are not parameter-matched, so an improvement cannot be solely attributed to covariance rather than added capacity. Any later network requires dimension/parameter/compute controls and original-module comparisons.

## Integrity and resume

Freeze driver/statistics/protocol/trainer/backbone hashes, original dataset config/manifest/cache metadata, paired fold source metadata/checkpoint/prediction hashes and projection SHA. Feature caches per source checkpoint under .cache_statistics_study (outside Git) must lock source/config signature and cache SHA. Fit heads saved with array SHA, training transform and coefficients; complete.json locks head/prediction hashes. Resume only valid signed units; never retrain CNNs. Thirty job units each extract once and fit three heads. Concurrency2, CPU threads2, DataLoader workers2.

Preflight: independent NumPy moment/covariance reference, constant/offset/permutation invariance as appropriate, projected covariance symmetry, exact feature dimensions; source model eval/forward probability reproduction; ridge coefficient/transform reproduction and fit-only moments; signature/source and projection checks. Labels never change. At completion audit every source checkpoint, feature-cache signature/hash, 90 head fit-index sets, 90 outer prediction sets and six schemes x3 pooled OOF CSVs with2386 rows each, patient isolation and login ownership.

## Analysis

Primary comparisons gap_cov vs gap within each encoder, with gap_std vs gap as simpler second-order controls (four paired comparisons total). Pool folds before per-seed macro-F1; report all seed differences, malignant recall and patient-mean accuracy. Use185 patient groups and20000 shared resamples. Report both usual95% intervals and Bonferroni simultaneous-family intervals using [.625,99.375] percentiles for four comparisons. Strong follow-up signal: all3 seed changes positive, mean>=+.01 and family interval lower>0; additionally require same positive mean direction in the other encoder stratum before investing in covariance-specific architecture. All weak/negative results retained. No test scoring, no grade weighting or automatic labels.

Comparing ridge GAP against original CNN head is descriptive only; changing the classifier is not evidence for second-order statistics. Repeated development and overlapping training folds limit these conditional intervals. This is an internal diagnostic, not a new clinical validation or new architecture contribution.

## Verified references

Lin, RoyChowdhury, Maji, ICCV2015: https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html . Bilinear/orderless feature interactions motivate the hypothesis only.
Li et al., ICCV2017: https://openaccess.thecvf.com/content_iccv_2017/html/Li_Is_Second-Order_Information_ICCV_2017_paper.html . MPN-COV is a related established method; this probe does not implement its matrix power normalization and must not be labeled as a faithful reproduction. Independent equations; no external code copied. No directly named covariance/bilinear entry was found in the shared library filename scan; do not invent a library provenance or novelty claim.

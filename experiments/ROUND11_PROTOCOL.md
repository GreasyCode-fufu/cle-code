# Round11 preregistration: patient OOF check of full DINOv2 adaptation

Round10 complete primary CLE structure gate failed. Candidate .875779, ordinary
local .875194, LoRA .872516, frozen head .865834, full fine-tuning .892762.
Large adapter effects do not imply predictive gain. Do not enumerate more similar
adapters from this negative result. Test full adaptation on patient cross-fitting
before proposing another structural contribution. This is a system-level internal
confirmation attempt, not a new architecture or an external cohort.

## Frozen design

Only 15 new neural models: full DINOv2 fine-tune, 5 outer folds times seeds
42/43/44. Reuse exact experiments/data_study/oof_jobs.json patient fit/inner/outer
splits. Train, checkpoint-select and score only original train indices0..2385
(2386 images,185 groups). Original fixed val538 and test671 never loaded or used
for fitting, checkpoint selection or scoring. No A–E grades/sample weights,
filtering, labels rewritten, pseudo-label supervision or new metadata targets.

Execute the unchanged round10 train_job with its unchanged model/optimizer/AMP
kernel, redirecting only experiment directories, job splits and verification.
Official public DINOv2 ViT-S/14 through the same declared timm1.0.27 port, same
source SHA, position interpolation, CLS+patch-mean768 linear3 head. All encoder
parameters trained. 60epoch, batch32, original resize256/crop224 augmentation and
normalization, AdamW wd.05 encoderlr1e-5/head1e-4, cosine0, AMP dynamic scaler,
gradient norm clip1 after unscale. Best checkpoint chosen exclusively by inner
patient validation, outer patients inferred only after checkpoint freeze.
Parallel2 and CPUthreads2. This round does not optimize learning rates or epochs.

The old input cache contains3595 all-split images, but loaders only access original
train rows. Do not claim the physical cache excludes val/test. No new image cache.

## Three frozen references, no retraining

1. resnet_bn_train: original 15 ordinary-BN ResNet34 OOF models, original CE heads,
   reproduced/cross-checked in round7. Exact same fit/inner/outer and seed.
2. resnet_bn_frozen: round7 15 frozen-BN ResNet34 OOF models, original CE heads,
   exact same fit/inner/outer and seed.
3. dino_frozen_ridge: round9 DINO CLS+patch-mean fixed-feature ridge OOF, exact same
   fit and outer rows, fit-only moments and alpha1, inner unused for head fitting.

Preserve source checkpoint/job/config signatures and reference CSV SHA. References
1/2 use different architectures/optimizers/gradient clipping/pretraining regimes
than new DINO; reference3 uses different head fitting/augmentation/normalization.
Thus differences compare complete pipelines, not a pure causal estimate of
self-supervision, backbone updating, architecture or fine-tuning alone. Public
pretraining overlap remains unaudited. Three seeds share one public initialization
but have paired split/head/training randomness; not three pretraining runs.

## Primary analysis and decision

Per algorithm and seed, merge all five outer folds to2386 predictions and THEN
compute macro-F1. Fold averages are descriptive. Each image gets exactly one
outer prediction per seed (three total). Report malignant recall, patient-equal
accuracy, class confusion, inner best versus last, source parameter RMS drift.
Three main comparisons: dino_full versus each reference. Shared bootstrap over
185 patient groups,20000 draws RNG20261024; average three seed differences.
Report ordinary95 and Bonferroni family interval [0.8333333333,99.1666666667]
percentiles for three comparisons. Strong internal follow-up requires all three
seed differences positive, mean>=1pt and family lower bound>0 in EACH comparison.
No changing primary reference, thresholds or grading based on interim results.

The same internal patients have supported earlier development, folds overlap in
training, and checkpoint choice/training variation are only partly reflected in
these conditional intervals. This is not external confirmation or a clinical
performance guarantee. CNN/classifier/augmentation confounds are reported.

If positive, establish an effective baseline then specify CLE adaptations and
faithful original-module, parameter/compute controls, followed by data-by-structure
tests. If weak/negative, investigate representation/learning evidence rather than
infinite module enumeration. Never claim negative structure trials prove label
noise or the impossibility of architecture improvement. Strict outer CV grading
must be nested or truly independent; final candidate frozen before one full test
audit with historical test use disclosed. No publication or novelty promise.

## Reproducibility and audit

Driver experiments/run_dino_oof_study.py; all earlier sources and protocols stay
unchanged. Freeze this protocol, immutable round10 training/model/loader/runtime
hashes, original folds, manifest and copied references. Same run signature and
checkpoint SHA guards. Worker must reuse a valid complete checkpoint on any
prediction/summary failure. No valid completed training repeated.

Before launch verify 15 exact splits, three-way patient isolation, onlytrain scope,
coverage, source/reference hashes, matched initialization, optimizer groups and
finite AMP update with correct clipping. At completion independently check all
0..59 logs, inner-best choice, checkpoint/key/parameter finiteness/source drift,
outer-only prediction coverage and hashes, references and all12 merged OOF files,
patient intervals/primary gate, login ownership. Save reports, code, complete runs,
reference indices and configurations to chat outputs; large weights stay server.

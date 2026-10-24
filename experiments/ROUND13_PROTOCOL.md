# Round13 preregistration: secondary fixed-fusion replication

R12 primary frozenBN-CNN+DINO gate FAILED. Prespecified secondary ordinaryCNN+DINO
scored .885805; its two prespecified comparisons versus DINO and ordinaryCNN had
positive family intervals and means+1.190/+2.676pt. This is NOT a replacement of
the failed primary. Replicate that secondary and add the missing same-family
ensemble comparator before any new CLE hybrid architecture. The same185 patients
are reused: new split/training randomness is not a new external patient cohort.

## New fixed repetitions and source training

Seeds45/46/47, chosen before any new model output. New StratifiedGroupKFold5 outer
and StratifiedGroupKFold10 inner (seed*100+fold), use the same first-valid-three-class
inner split rule as original DATA. All original train2386/185 groups, no fixed
val538/test671 access, grades/weights/filtering/relabeling/pseudo labels. Validate
fit/inner/outer patient disjointness and each seed outer coverage2386. Copy original
manifest and freeze every exact new split before training. Do not search seeds.

45 NEW neural models:3source algorithms×3seeds×5folds. No valid existing source
model has these new held-out sets, so these are necessary new trainings. CNNs:
ordinaryBN and frozen-running-statistics BN ResNet34, all affine parameters train.
Use unchanged original core.run_worker for both with original pretrained weights,
60epoch/batch32/AdamW1e-4wd.05/cosine0/AMP/noearlystop, no added clipping.
FrozenBN train override matches the audited R7 policy; only BN buffers frozen.
DINO: unchanged R10 full_finetune train_job, public ViT-S/14 same timm port/CLS+mean
head,60ep/batch32/encoder1e-5/head1e-4/wd.05/cosine0/AMP/unscale clip1.
Source recipes deliberately reproduce R12 pipelines, not a claim to isolate
architecture/self-supervision causally. Record parameters/FLOPs; not compute matched.

Inner patients choose checkpoint; outer patients inferred only after freeze.
Original resize256/crop224 augmentation and ImageNet normalization, evalcenter224.
Physical inputcache3595 all splits unchanged, effective loaders only originaltrain.
Parallel2/CPUthreads2. Public pretraining shared across seeds; heads/training/folds
vary as specified. No new public downloads or dependency installations.

## Unchanged scalar calibration and six pipelines

For each new best source, eval/freeze and extract inner/outer AMP logits asFP32,
state unchanged. Reuse immutable R12 prediction_fusion.py: beta[.05,20] fit inner
NLL only, same bounded convex optimizer/endpoints and original-label supervision.
Save signed raw/calibrated outer predictions/logits/temperature. Positivebeta
preserves single-source argmax; outer labels never fit temperature or fusion.
Inner also chose checkpoint, not independent calibration cohort.

Same six derived pipelines:ordinaryCNN,frozenBN-CNN,DINO,ordinaryCNN+DINO,
frozenBN-CNN+DINO,ordinaryCNN+frozenBN-CNN. Every pair uses exact same seed/fold;
equal.5 fixed probabilities, no gating/weight sweep/source swapping/seed ensembling.
Primary now declared BEFORE new outputs:ordinaryCNN+DINO. This explicitly
replicates a secondary finding; R12's primary failure remains in the record.

## Analysis and gate

18 mergedOOF files (6schemes×3seeds), each2386; merge5folds THEN macroF1.
Report all seeds, malignant recall, patient-equal accuracy,NLL/Brier, error overlap,
source inner best/last epochs and source-state/BN behavior. Primary comparison
family ONLY3:ordinaryCNN+DINO versus ordinaryCNN,DINO,andCNN+CNN. Use185patient
20000sharedbootstrap RNG20261024,95 andBonferroni family[2.5/3,100-2.5/3] percentile
intervals. Strong replication requires EACH comparison:3positive seed deltas,
mean>=1pt,familylower>0. No threshold relaxation or changing primary again.
Other derived pipelines descriptive, cannot replace the primary.

The repeated internal patients, overlapping fit sets, development history and
training/pretraining variation limit conditional intervals; no external clinical
claim. If replicated, design a CLE hybrid/efficient distillation mechanism with
faithful original methods, paired initialization, retrained ablations and budget
controls, then patient confirmation/data-by-structure. Fusion itself is mature,
not publication novelty. If not replicated, report instability and use evidence
to change mechanism, not sweep mixture weights or similar modules endlessly.
Nested CV grades must be computed inside outer train or truly independent.
Final candidate frozen before one full test audit with historical test use disclosed.

## Reproducibility and safe resume

Driver run_fusion_confirmation.py, directoryexperiments/fusion_confirmation,
weights/fusion_confirmation, ignored.cache_fusion_confirmation. All old code and
configs untouched; dependencies hash original core/R10 trainer/model/DINO loader,
R7 BN policy andR12 helper/runtime. Freeze new source/config/job/split hashes.
Each source keeps60ep log/run signature/bestcheckpointSHA. Resume valid source
checkpoint on calibration/prediction failure; no completed neural model retraining.
R13 worker has custom outer_complete with raw/calibration/temperature/cache SHA;
core complete remains source-training metadata (CNN may also have outer_metrics).

Preflight checks exact new splits/coverage/class isolation, paired CNN init and
BN counters/affine gradients, finite AMP updates/source optimizer groups, budgets
and pretrained/runtime/cacheSHA. R12 calibration algebra is unchanged, reuse its
audited helper rather than invent new calibration method. Completion independent
audit:45 new models/60ep/inner-onlybest/source drift/frozenBNbuffers, full logits
and calibration reproduction,18OOF coverage/metrics,3familycomparisons/gate,
loginownership. Save reports/code/config/splits/all45runs and18OOF tochatoutputs;
weights/logit caches remain server. Every effective change/result Git committed.

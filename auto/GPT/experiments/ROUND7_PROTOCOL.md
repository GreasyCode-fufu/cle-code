# Round 7: patient-OOF check of the weak BN-statistics signal

Preregistered 2026-10-24, before new OOF training. This is an internal measurement/normalization study, not a new architecture or independent external validation.

## Evidence and question

Round 6 frozen-BN grade-weighted mean val F1 .889178 versus .881958 for standard BN. Paired differences +.015708, +.009357, -.003403; patient interval [-.008611,.022902]. It failed the predefined promotion gate. Head-only .845186 and starting-point anchor .882276 did not improve reliably. Do not treat the weak signal as established and then design modules around it.

Question: does freezing pretrained BN running statistics improve full-data CE patient-OOF generalization, when grading and the reused fixed validation set do not enter this comparison?

## Design and reuse

Two algorithms x seeds42/43/44 x five outer patient folds =30 compared models. Reuse the original 15 plain ResNet34 full-CE OOF models from experiments/data_study/oof_jobs.json. Verify original job/config/trainer signatures, checkpoint SHA256, 60-epoch records, outer prediction identity and metrics before reuse. Only 15 frozen-BN models are newly trained. Reuse original per-seed outer folds and their inner patient splits exactly; do not regenerate folds, retrain valid baselines or choose a favorable partition.

Only original train2386 / 185 leakage groups participate. Original val538 and test671 are not scored or used for selection/training in this round. The existing label-independent 256 cache contains all original splits, but effective loaders must access only original-train indices. Do not claim that this cache excludes test. Source-image labels, IDs and patient groups are unchanged.

Outer scoring patients are disjoint from both fit and inner checkpoint-selection patients. Each image has exactly one outer prediction per algorithm per seed, hence three predictions per algorithm. All three classes must exist in fit/inner/outer. Same 60ep/AdamW1e-4 wd.05/cosine0/AMP/batch32/original256-to224 augmentation, pretrained checkpoint and paired classifier initialization. No early stopping. Concurrency2 / CPU threads2.

No A–E grades or weights are used. Reusing old global grades inside outer training would risk leakage from outer patients. This full-CE study isolates BN behavior without that issue; it does not by itself estimate BN x grade-weighting interaction.

New models use the unchanged core.run_worker training implementation. An instance train(mode) override only freezes BatchNorm2d running statistics whenever training mode is requested, while every parameter including BN affine remains trainable. Eval is normal. No additional inference parameters, images, metadata or modules. This is a mature BN control, not a novel CLE network.

## Before launch and completion audit

Freeze hashes of driver/protocol/core/supervisor/backbone, original OOF jobs/data config, manifest/cache metadata and copied baseline artifacts. Independently check baseline outer predictions exactly match source folds, normalized finite probabilities, labels/groups and completion metrics; all fit/select/score patient sets are disjoint and cover all train indices exactly once per seed. Verify cache full SHA256 once before launch.

Check paired eval logits bitwise identical, parameter/state key counts unchanged, all parameters trainable, training call freezes all 36 BN counters and preserves affine gradients, finite real-image AMP step at a settled scale; normal dynamic scaler remains in actual training. Confirm mode False turns all submodules off. Training implementation is the exact immutable original core file.

At completion verify15 new +15 reused checkpoint hashes/signatures, all60epoch logs, frozen BN buffers equal initial pretrained values and all parameters finite. Verify outer holdout never used for checkpoint selection. Aggregate full2386 OOF rows per algorithm/seed, each image three times per algorithm, patient isolation, input range and login ownership. Preserve all fold metrics and predictions.

## Analysis and decisions

Primary metric pooled-OOF macro-F1 per seed (pool five folds before computing F1), then paired seed mean. Fold means are descriptive. Report malignant recall and patient-mean accuracy. Bootstrap 185 patient groups with 20,000 shared resamples and average the three paired seed differences inside each resample. Confidence intervals condition on fitted cross-fitted models; overlapping training folds, development selection and training variability are not fully represented. This is internal OOF replication, not a new cohort or unselected confirmatory p-value.

Strong diagnostic signal: all three paired seed differences positive, mean >=+.01 and patient-cluster95 lower >0. Weaker changes are saved as unresolved. No new structural claim follows from this result. If positive, next separate data x BN interaction and normalization-aware architecture work using leakage-safe grading and matched controls. If negative/unresolved, do not blindly enumerate normalization modules or claim that all architectural research failed; reassess measurement and representation evidence. No test evaluation or relabeling in this stage.

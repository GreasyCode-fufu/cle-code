# Round 9: frozen independent-pretraining representation baselines

Preregistered before extracting CLE features or fitting new heads. This is baseline research, not a new CLE network or a controlled causal study of self-supervision.

## Evidence and question

Round8 fixed second-order probes did not improve reliably: std/cov differences -.30 to +.07pt; all ordinary and family intervals cross zero. Source probabilities and90 convex ridge optima audited. Do not blindly train covariance modules. Earlier full finetuning overfits while frozen ImageNet ResNet34 head-only underfits under the fixed recipe. Question: is a different externally pretrained representation a stronger starting point than this ImageNet CNN, before proposing task-specific modules?

## Models and provenance

1. Existing ImageNet pretrained ResNet34, frozen, GAP512. Use exact approved checkpoint SHA333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef; its randomly initialized3-class fc is unused.
2. DINOv2 ViT-S/14 without registers, publicly available Meta weights, through installed timm1.0.27 port. Official URL https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth . Downloaded88,283,115 bytes; experiment-locked SHA256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 (computed here, not a publisher signature). No additional downloads during extraction/fitting.

Official model card states Apache2.0, LVD-142M pretraining,384 features and one CLS plus256 patch tokens at224. Verified Git blobs: LICENSE5471dc10377b76db85a2feca7a99a7eef4980ba8; MODEL_CARD21b9bf295c8cab14e782e1a7a1d051be9e501088; hub/backbones.py af5fd95d38eb33f73a85b92a94a715773a4e70bc. These are blob identifiers, not commit IDs. License included. Sources: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md and https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py . No claim of being the latest or best foundation model.

Load all timm inference keys strictly. Only remove unused mask_token and bicubic-resample pos_embed from1370 to257 tokens, antialias=False. Every other weight must remain exactly identical. This is the specified timm port at224, not a claim of exact official-repository inference: timm size-based position interpolation differs from the official repository's offset interpolation. Freeze installed implementation hashes/version. No learned data-specific initialization or network updates.

## Fixed data and heads

Only original train2386/185 patient groups, original per-seed5-fold fit/inner/outer splits; no new partition. No A–E grades, fixed val/test loading, relabeling or sample removal. Same original resize256/center224 and ImageNet RGB normalization. Both frozen encoders in eval, all parameters requires_grad=False; extract FP32 with TF32 disabled. Extract each independent pretrained encoder once for all2386 images; imagewise features never estimate a dataset statistic and never consume labels. Cache includes indices0..2385 only. Original shared input cache contains other splits, but loaders never access them.

Four new schemes: resnet_frozen_gap512; dino_cls384; dino_mean384 (mean of normalized patch tokens); dino_concat768 (CLS and patch mean as two blocks). CLS+mean is a conventional output choice in the official model card, not our invention. Primary candidate fixed in advance: dino_concat. Others are representation/pooling controls, not a post-hoc best-of-three primary.

Each of15 fold units fits4 Ridge heads,60 analytic fits, zero new neural trainings. Use exactly original fit patients only; inner patients unused for head fitting/tuning. Per-feature means/std fit-only; floor1e-6; scale block by sqrt(block_width * number_of_blocks), Ridge alpha1/intercept/cholesky, onehot original labels. Same head recipe as R8. Softmax ridge scores uncalibrated. Three seeds are the existing patient split replicates, not three independent public pretraining initializations; frozen feature array is common across them.

Copy two previously audited R8 GAP-ridge OOF reference schemes: standard-BN CLE-finetuned ResNet and frozen-BN CLE-finetuned ResNet (six pooled CSVs, no refitting). Six total comparison schemes. Pretraining data, architecture, precision and compute differ; dimensions also differ. Report them and avoid attributing a gain solely to self-supervision, architecture or a new module. Parameter counts/runtime recorded, no compute-matched claim. R9 does not test DINO finetuning, other resolutions or the maximum possible linear-probe performance. Potential overlap of public pretraining corpus with medical imagery is not audited; no new-cohort claim.

## Verification and resume

Freeze source/runtime/protocol/license/weight hashes, data/OOF jobs/manifest/cache metadata and copied reference hashes. Before launch verify full original cacheSHA, all15 three-way patient splits and coverage, strict DINO weights/key mapping, declared position interpolation against independent F.interpolate, CLS output agreement, finite features and exact token dimensions, all encoders' states unchanged by inference, fit-only transforms and independent ridge optimum. Deterministic FP32 re-extraction of selected rows must agree within declared numerical tolerance; no feature-label tuning.

Global feature cache .cache_representation_study/features.npz (2386x1280:ResNet512/CLS384/mean384) with signatureSHA and indices. Head arrays under weights/representation_study and signed completion artifacts in experiments/representation_study. Reuse valid feature cache and completed head units, never retrain source models. Concurrency2, CPU threads2. Audit60 fit-index sets/transforms/ridge normal equations, outer predictions,18 mergedOOF CSVs (6schemes×3) each2386, reference identities, patient isolation, source hashes and login ownership.

## Decisions

Per-seed metric pools five outer folds before macro-F1. Also malignant recall/patient-mean accuracy. Five predefined comparisons: DINO cls/mean/concat each vs frozen pretrained ResNet; DINO concat vs each of the two R8 finetuned GAP-ridge references.185 patient groups,20000 shared resamples, ordinary95 and Bonferroni family intervals [.5,99.5] percentiles. Strong follow-up requires dino_concat all3 seed differences positive, mean>=+.01 and family lower>0 against frozen pretrained ResNet AND both prior finetuned references. Save weaker/secondary results without changing this gate. Secondary-head gains need a separate preregistered replication, not relabeling the primary.

Internal development-selected, conditional cross-fit intervals do not cover repeated selection/overlapping fit sets/training uncertainty. A strong external representation is a foundation for later CLE adaptation, not proof of new structure. Any task-specific adapter must subsequently beat faithful pretrained baselines and parameter/compute controls, with leakage-safe grading if used. No test evaluation until a final candidate is frozen, and acknowledge test was used historically.

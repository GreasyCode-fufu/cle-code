# Round 6: constrain pretrained-backbone adaptation

Frozen before training, 2026-10-24. This is a transfer/generalization diagnosis, not a claim of a new CLE architecture.

## Motivation and hypothesis

Four structural rounds showed no confirmed additional contribution. Round 5 GCE and ELR did not beat grade-weighted CE. Its final training accuracy was 99.0–100%, while all final validation F1 values were below the best selected epoch. This is evidence to test constrained adaptation, not proof of label noise, catastrophic forgetting or a causal explanation of generalization error.

Hypothesis: retaining pretrained representations or limiting trainable capacity improves patient-generalization beyond the existing grade-weighted, fully finetuned ResNet34. Three distinct constraints and a BN-statistics control separate parameters, adaptation depth and running-statistic effects.

## Arms and frozen recipe

12 new trainings: four arms x seeds 42/43/44. Reuse six verified full_ce and grade_ce references from round 5 without retraining or additional inference. New arms all use the exact grade_ce sample weights (within-original-class normalization), original 256 cache / 224 augmentation, original train 2386 and full val 538 / 39 groups, pretrained checkpoint SHA, paired head initialization, 60 epochs, batch32, AdamW lr1e-4/wd.05, cosine-to-zero, AMP, CPU threads2, concurrency2.

1. head_only: train fc only; backbone parameters and all backbone BN running statistics frozen.
2. stage4_only: train layer4 + fc; stem/layer1–3 parameters and their BN statistics frozen; layer4 BN trainable normally.
3. bn_frozen: train all parameters including BN affine; freeze all BN running statistics. Same trainable count as full fine-tuning.
4. starting_anchor: train all parameters; add .01/2 times summed squared distance of every non-fc parameter from its pretrained initial value. Keep original AdamW unchanged. This is an L2-SP-inspired additional constraint, not a strict reproduction of the authors' optimization recipe. No coefficient search.

All four retain the same inference architecture and input. Frozen-head/partial-update results are conditional on this single common optimization recipe, not a claim of globally optimized linear-probe performance. Labels are unchanged; no samples, val or test are filtered. No test evaluation. No extra metadata or grading supplied to inference.

## Validation and diagnostics

Before launch: verify copied baseline file hashes, six checkpoint signatures/SHA, 60-epoch records; paired initialized eval predictions identical; each policy's trainable names; finite real-image AMP gradient/update; frozen parameters and frozen BN counters remain exactly unchanged; trainable fc/layer4 gradients nonzero; anchor initially zero, finite, and analytic gradient equal to .01*(parameter-initial). Evaluate policy models in normal eval mode.

Record trainable parameter counts, final/best epoch, CE and anchor contribution separately, non-fc parameter RMS drift, full val predictions. At completion independently verify all 12 source/checkpoint signatures, 60 epochs, full val indices/probabilities/metrics, patient-disjointness and login ownership. Verify frozen parameter equality to initialization and correct BN counter behavior in best checkpoint; pretrained running counters may be nonzero, so compare against initial values rather than zero.

Primary outcome mean macro-F1 over the paired seeds. Report patient-cluster paired 95% intervals with 20,000 shared resamples, malignant recall and patient-mean accuracy. Strong signal requires three positive seed differences, mean >=+1.0pt vs grade_ce and interval lower >0; also mean >0 vs full_ce. All arms and failed hypotheses retained. Multiple arms can satisfy; choose largest development mean only for an explicitly exploratory next confirmation. No gains are attributable to architecture novelty in this round.

Intervals condition on fitted models, do not cover training/selection uncertainty, and are uncorrected for repeated development selection/multiplicity. A positive result requires patient-level confirmation with leakage-safe grading before any new architecture claim. Strict outer CV grades must be derived inside outer train or from a truly independent grader. Historical test has been used already; a final frozen candidate would receive one further clearly labeled audit, not an external-validation claim.

## Source

Li, Grandvalet, Davoine. Explicit Inductive Bias for Transfer Learning with Convolutional Networks, ICML 2018. Formula (2)/(3) inspires the starting-point penalty. https://proceedings.mlr.press/v80/li18a.html . Independently implemented equations; no copied external code. Mature freezing, BN and L2-SP ideas cannot be presented as new modules. The shared module library was rechecked for adaptation/LoRA entries; its filename claims are not evidence of publication or suitability. No library component is blindly added in this prerequisite study.

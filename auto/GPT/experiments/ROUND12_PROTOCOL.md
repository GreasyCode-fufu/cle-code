# Round12 preregistration: fixed CNN/ViT complementarity diagnosis

Round11 full DINO OOF .873909 matches frozen-DINO ridge .873057; its complete
three-reference gate FAILED. No confirmed new CLE module. Stop enumerating similar
adapters. Test whether differently trained CNN and ViT errors support a future
hybrid design. This is a fixed ensemble/calibration diagnostic, not new architecture.

## Scope and source units

Zero new neural training. Reuse45 audited source checkpoints:15 ordinary-BN CNN
from original DATA,15 frozen-BN CNN from round7,15 full-finetuned DINO from round11.
Each source retains its exact original seed42/43/44 fit/inner/outer patient split.
One inference/calibration unit per source:45 units,45 scalar temperature fits.
Only original train2386/185 groups. Fixed val/test not loaded; no grades, filtering,
relabeling or pseudo-label supervision. Physical original cache contains3595 all
splits, effective loaders only use each source's inner/outer indices0..2385.

Source checkpoint already selected using inner patients. Extract inner and outer
logits at original center224/resize256 and original batch32, AMP forward then FP32
logits. Reproduce original raw outer probabilities within2e-6 and same argmax.
Every source is eval/requires_grad=False; state SHA before/after must agree.
Store effective indices/logits in signed cache; reuse valid caches/units.

## Inner-only calibration and fixed six pipelines

Fit one inverse temperature beta in [.05,20] by minimizing original-label mean
cross entropy(logits*beta) on INNER patients only. Positive beta preserves each
source's class decision. Use scipy bounded scalar convex-in-beta optimization,
xatol1e-10/maxiter500; compare objective to beta1 and endpoints for numeric sanity.
Record beta,T=1/beta,inner NLL before/after,source/checkpoint/cache SHA. Outer
labels never enter calibration or weighting. Inner was also used for checkpoint
selection: not an independent calibration cohort; outer results remain the check.
No bound search, method search or outer tuning. This is established temperature
scaling, independently implemented from the equations, not claimed novel.

Six predefined pipelines for EACH original outer fold:
1. resnet_bn_train: calibrated ordinary CNN.
2. resnet_bn_frozen: calibrated frozen-BN CNN.
3. dino_full: calibrated full DINO.
4. mix_train_dino: .5 ordinaryCNN probability + .5 DINO probability.
5. mix_frozen_dino: .5 frozenCNN + .5 DINO (PRIMARY fixed candidate).
6. mix_resnets: .5 ordinaryCNN + .5 frozenCNN (same-family ensemble control).

Mix only source models with exactly same seed/fold and fit/inner/outer sets.
No learned gating/weights, oracle selection, seed ensembling or source swapping.
Single-source argmax must equal its old raw pipeline; only probabilities change.
At inference, mixtures require both source networks, with additional parameters/
computation; not equal-budget/zero-cost/new-network evidence. The same-family
control is a mechanism diagnostic, not a precise compute-matched alternative.

## Analysis and frozen decision

Six pipelines×3seeds=18 merged OOF files, each2386 rows. Merge5folds THEN score
macro-F1, not average foldF1. Each image has3 held-out predictions per pipeline.
Report malignant recall, patient-equal accuracy, NLL, multiclass Brier score,
source error overlap and disagreements. Positive scalar calibration may help or
hurt held-out NLL and can never itself improve single-source classificationF1.
Do not describe fitted probabilities as clinically validated confidence or A–E.

Five comparison family: mix_frozen_dino versus dino_full, resnet_bn_frozen and
mix_resnets; secondary mix_train_dino versus dino_full and resnet_bn_train.
185-patient20000 shared bootstrap RNG20261024, average3pairedseed deltas;95 and
Bonferroni family99 [.5,99.5] percentile intervals. Strong PRIMARY follow-up needs
all3pairedseed deltas positive, mean>=1pt, family lower>0 in EACH of its3comparisons.
Secondary cannot replace primary. Do not tune thresholds or fusion if intermediate
results disappoint. Intervals condition on cross-fitted models and do not cover
all overlapping-training, repeated-development or training/pretraining uncertainty.

If strong, preregister CLE hybrid training with faithful original methods, paired
source initialization, ablation and parameter/compute controls; then independent
patient confirmation/data-by-structure. If weak, use error evidence to choose a
different mechanism rather than blindly scaling fusion variants. Do not claim
ensemble success equals architecture innovation or label-error discovery. Strict
outer CV grades must be nested or truly independent. No test until final candidate
freeze and one complete audit with historical use disclosed. No paper guarantee.

## Reproducibility

New driver experiments/run_fusion_study.py; helper experiments/prediction_fusion.py;
all prior sources/config/protocols unchanged. Freeze45 source jobs/configs/signatures,
checkpoint SHA, split manifests, runtime, helper/protocol/driver and job hashes.
Cached logits .cache_fusion_study/<unit>.npz/json; complete.json locks cache,
temperature.json and outer probability CSV. Parallel2/CPUthreads2.

Preflight source provenance/strict source reload/unchanged eval state, raw probability
reproduction, exact paired patient splits/coverage, convex calibration gradient,
positive-temperature decision invariance, source/mixture analytic reference,
only-inner labels passed to fit and effective-index range. Completion audit must
independently reproduce all45 temperatures/derivative optimum or bound condition,
source logits/probabilities/state integrity, all18 merged OOF/mixtures/metrics,
five comparisons/complete gate, and login ownership. No CNN or ViT is retrained.

Temperature scaling primary source: Guo et al., ICML2017, On Calibration of Modern
Neural Networks: https://proceedings.mlr.press/v70/guo17a.html . This experiment
specifies bounded inverse-temperature fitting and nested use explicitly; not a
claim to reproduce the paper's entire protocol or clinical calibration validity.

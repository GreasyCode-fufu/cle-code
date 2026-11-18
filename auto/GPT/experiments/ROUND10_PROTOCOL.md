# Round10 preregistration: controlled CLE token adaptation

Exploratory follow-up to round9, whose complete primary gate FAILED. Frozen
DINOv2 matches the best existing fine-tuned CNN reference but does not reliably
beat it. Test whether CLE domain adaptation helps before attributing gains to
new modules. No claim that label errors are the dominant limitation.

## Data and training

Original full train2386/185 patient groups and full val538/39 groups. Seeds
42/43/44, six variants, 18 new neural trainings. No grades/sample weights, no
relabeling, no test loading or scoring. Prior ResNet-derived global grades are
not used to select DINO training images. All original data/previous sources stay
unchanged. Original resize256/224 crop, H/V flip, rotate15; eval center224;
ImageNet normalization. Cache includes all3595 original images, while effective
loaders for this round only access train and val indices.

DINOv2 ViT-S/14 official public weights via the SAME frozen round9 timm1.0.27
port and documented position interpolation. Existing weight SHA and runtime
locked; no redownload/new dependency. CLS+patch-mean 768 descriptor and identical
new linear3-class head initialized before adapters in every paired seed.

60 epochs, batch32, AdamW weight decay .05, cosine to0, no early stopping.
Head/adapter lr1e-4; full-finetune encoder lr1e-5 and head1e-4. Clip trainable
gradient norm to1 after AMP unscale, applied to every arm. Original dynamic
GradScaler in training; preflight may use settled128 for finite-gradient checks.
Best checkpoint selected by full val macro-F1; this makes val an exploratory
selection set. Independent patient OOF confirmation is required for claims.
Parallel2, CPU threads2; no other agents' files/processes touched.

## Six paired arms

1. frozen_head: encoder fixed; augmented-image CE training of the linear head.
2. full_finetune: all encoder parameters trained, same linear head.
3. lora_qv: rank24, alpha24 Q/V low-rank updates in last four blocks; base fixed.
   A Kaiming, B zero. Faithful LoRA update algebra, independent implementation;
   not a reproduction of the original language-model performance experiment.
4. local_adapter: after each last-four block, non-affine LN -> down384-to48 ->
   GELU -> spatial depthwise3x3 -> GELU -> up48-to384, residual on patch tokens.
   CLS has no direct adapter delta. Generic convolutional-bypass control inspired
   by Convpass, not a faithful Convpass implementation.
5. cle_cls_context (preselected main candidate): same local pathway rank24;
   normalized CLS goes through linear384-to48 and splits into a,b. Before final
   GELU/up, local := local*(1+.5*tanh(a))+.5*tanh(b). Context scale is (.5,1.5),
   shift(-.5,.5). All source parameters fixed.
6. cle_mean_context: exact same parameter budget/pathway as5, context from mean
   normalized patch tokens instead of CLS. Tests the value of pretrained CLS
   context, rather than context conditioning versus a missing parameter block.

Local up and context linear zero initialized; LoRA B zero. Exact epoch0 identity
is checked. This uses a trainable output projection without an extra zero scalar
gate: output projection learns first, then lower/context layers receive gradients.
Preflight explicitly verifies this multi-step gradient sequence; identity alone
does not establish useful learning. Non-full encoders remain eval; new modules
are trainable and gradient propagation through frozen matrices is retained.

LoRA/local/context trainable budgets approximately150k, within3%; head2307.
Full source encoder ~21.6M. Record total/trainable parameters, forward counted
Conv/Linear+attention matmul FLOPs (2/MAC), actual AMP batch32 peak memory.
These controls have similar whole-network counted FLOPs, not identical branch
computations or latency. No zero-cost inference claim.

## Hypotheses, uncertainty, progression

Primary candidate versus the other five arms is the fixed comparison family.
39 patient groups, 20000 shared paired bootstrap draws; compare average of
three seed macro-F1 deltas; ordinary95 and Bonferroni family99 intervals
([.5,99.5] percentiles). A strong exploratory structure signal requires all three
deltas positive and mean>=1pt versus frozen_head; mean>=.5pt versus each of
LoRA and generic local control; mean>=.3pt versus mean-context; mean>0 versus
full fine-tuning; family interval lower bound>0 in ALL five comparisons.
Report every seed/variant, malignant recall, patient-equal accuracy and final
versus best val. No replacing primary candidate or relaxing thresholds.

Record same-checkpoint adapter-off descriptor/logit changes and disagreements,
per-image context summaries. These diagnose actual branch effect, not ROI truth,
retrained ablations or a noise diagnosis. Gradient/magnitude is not clinical merit.
Family correction covers only five within-round comparisons; repeated development,
checkpoint selection and training variation remain limitations. A positive result
requires new patient-level OOF confirmation, fair original methods/budget controls,
then data-by-structure interaction. Nested grades must be recomputed within each
outer training set. Final candidate frozen before one complete test audit, with
historical use of the existing test disclosed. No paper acceptance/novelty promise.

## Provenance and research positioning

- DINOv2 official: https://github.com/facebookresearch/dinov2 (Apache2.0;
  license retained in experiments/DINOV2_LICENSE.txt). Unchanged round9 loader.
- LoRA: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
  source blob637865600fc9afbf76831e99e3d71eaca6871bda; independently implemented
  low-rank equations, no copied source. Hu et al., ICLR2022.
- Convolutional bypass related work: https://github.com/JieShibo/PETL-ViT and
  https://arxiv.org/abs/2207.07039. Our simple post-block control differs in
  location, activation, token treatment and scaling; do not label exact Convpass.
- FiLM: https://ojs.aaai.org/index.php/AAAI/article/view/11671 (Perez et al.,2018).
  Bounded affine conditioning is established; not itself a new contribution.
- shared/module_library informed prior four architecture rounds. This round
  implements explicit token-adaptation equations; no invented library provenance.

Save immutable config/source hashes, run signatures, checkpoint SHA, full538
per-image predictions and logs0..59. Resume only validated completed models;
failure in prediction/diagnostics must reuse valid trained checkpoints. Independently
audit source freezing/head pairing/parameters/gradients/shapes/compute, complete
patient splits and results before interpreting this round.

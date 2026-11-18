# MTRD Implementation Report

## Outcome

Implementation status: **SUCCESS**. This phase implemented and verified Multi-step Transition Relation Distillation without starting a 300-epoch run.

## Architecture choice

`MarkovKDDetectionTrainer` subclasses the verified `KDDetectionTrainer`. It replaces only the loss construction and epoch metric schema. The original Gram `ChannelRelationKDLoss`, `KDDetectionTrainer`, and `scripts/distill_train.py` remain byte-for-byte unchanged.

The MTRD objective builds only `[B,C,C]` channel transition matrices, masks self-similarity before softmax, supports fixed self-loop probability, computes general positive transition powers with `torch.bmm`, and applies row-wise `KL(P_T^k || P_S^k)` in float32. It has no trainable parameters and does not change Student inference.

## Verification

- MTRD unit tests: 30/30 passed.
- Final combined MTRD tests: 31/31 passed, including real pipeline backward and standalone checkpoint inference.
- Real pipeline: Teacher/Student P3/P4/P5 shapes matched; GradScaler backward and optimizer step succeeded; Student updated; Teacher remained frozen and isolated.
- Gram RelKD regression: 11/11 passed; fixed-feature total and per-scale loss were bitwise identical to the pre-MTRD snapshot.
- Ultralytics version: 8.4.84.
- Core source and all protected historical weights/results: unchanged by SHA256.

## Temperature calibration

Only 20 real training batches were used; no validation/test metric, backward, or optimizer step participated in selection.

- Recommended temperature: **0.05**
- Selection status: **PASS**
- P3 entropy median: 0.36487016
- P4 entropy median: 0.33952019
- P5 entropy median: 0.38105440

## Weight calibration

- Median raw MTRD/detection ratio: 0.4879813813
- 2.5% weight: 0.05123146283240
- 5% weight: **0.10246292566479759**
- 10% weight: 0.20492585132960
- Student state restored after calibration: True

## Gradient and redundancy diagnostic

| Scale | Weighted MTRD/Detection grad ratio | MTRD-vs-Gram cosine |
|---|---:|---:|
| P3 | 0.13600727 | 0.26974148 |
| P4 | 0.29307632 | 0.26991680 |
| P5 | 0.65810820 | 0.25143373 |

P5 exceeds the 0.5 warning threshold, so the 5% candidate carries a **gradient-scale WARNING**. The weight was not changed automatically. Cosines near 0.25–0.27 indicate a distinct gradient signal in this five-batch diagnostic; this is not evidence of long-term behavior or accuracy superiority.

## Smoke test

- Status: SUCCESS
- Epochs: 1
- Validation executed: False
- Finite detection/MTRD step losses: True
- Sustained batch memory growth: False
- Peak CUDA memory: 565.777 MiB
- Teacher unchanged/eval/isolated: true
- EMA updated and Student changed: true
- Hooks removed and CUDA cache released: true

## Recommendation boundary

The calibrated 5% weight is a first formal candidate, not a claim that MTRD will outperform Gram RelKD. Because P5 gradient ratio triggered WARNING, formal training should preserve logging and closely inspect early-epoch P5 behavior. This phase stops here and did not launch 300 epochs.

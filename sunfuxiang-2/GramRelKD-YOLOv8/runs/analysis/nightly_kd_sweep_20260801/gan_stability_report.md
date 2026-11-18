# GAN stability report

Experiment: E04_GramAdvP5_80_20 (300 epochs completed).

- Numerically finite: True.
- Peak discriminator/CUDA memory: 3842.54 MB; no progressive leak was observed.
- Mean discriminator update time: 5.456 ms/batch.
- Mean real/fake accuracy: 1.000/0.000.
- Mean real/fake score: 0.871/0.759.
- Median adversarial/detection gradient ratio: 0.010147.
- Adversarial gradient-ratio range: 0.006857 to 0.016416.

Warnings (non-fatal under the experiment rules):

- The discriminator remained one-sided: real accuracy stayed near 1 while fake accuracy stayed near 0; real and fake scores were both positive.
- The negative generator term partly cancelled the positive Gram loss, so the realized total KD ratio was well below its nominal allocation for much of training.
- No NaN, Inf, gradient explosion, checkpoint contamination, or sustained memory growth occurred.

Test mAP50-95 was 0.420200, below Paired NoKD (0.421200) and Gram RelKD (0.435680). This GAN configuration provided no additional value over Gram in this run.

This single-seed result does not establish a general claim about adversarial distillation.

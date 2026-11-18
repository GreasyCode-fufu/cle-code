# Formal MTRD training report

Training completed all 300 epochs in 3662.48 s (entry elapsed 3679.40 s). The best checkpoint is epoch 222, with validation mAP50-95 0.432620.

The Student initialization hash is `b5c987921cb2eae9d9536c8c632c619145e7d56a992f37d9badff8a368ae25b6` and matches the Gram RelKD/Paired NoKD random-YAML initialization: **True**. Temperature, KD weight, layer/step weights, self-loop probability and warmup remained fixed.

Post-warmup MTRD/detection ratio median is 3.38%; mean is 3.36%. All step losses remained finite. Entropy non-collapse: **True**. P5 abnormal dominance: **False**.

MTRD raw loss changed from 1.695792 at warmup completion to 1.190364 at epoch 290 (-29.80%), then to 1.344859 at epoch 300 after close-mosaic (12.98%). Thus the long-run trend declined, with an expected distribution shift in the final no-mosaic window rather than monotonic per-epoch decay.

Teacher forward averaged 9.6719 ms/batch and transition loss 5.9458 ms/batch. Training time versus Paired NoKD changed by 4.95% and versus Gram RelKD by 4.50%. Peak CUDA memory was 3840.62 MiB.

The exact-once feature-hook contract was checked on every KD batch by `FeatureHookBank.validate_once()`; the completed run had no hook exception. Sustained batch-memory growth: **False**.

See `mtrd_training_summary.csv` for all requested epoch nodes and `transition_stability_report.md` for close-mosaic and step-level analysis.

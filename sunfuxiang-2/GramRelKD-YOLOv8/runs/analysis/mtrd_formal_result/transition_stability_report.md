# MTRD transition stability report

- Post-warmup ratio (epochs 11–300): mean 3.36%, median 3.38%, range 2.48%–5.00%.
- All normalized transition entropies stayed above 0.2: **True**. Final P3/P4/P5: 0.338282/0.328233/0.367381.
- Step2 collapsed near zero: P3=False, P4=False, P5=False.
- P5 post-warmup total-loss dominance ratio: 1.2281 (abnormal threshold 2×: False).
- Epoch10→300 step reductions: P3 28.65%/12.48%; P4 30.14%/20.20%; P5 16.52%/11.82%.
- Close-mosaic (291–300 versus 281–290) raw-loss change: 14.03%; ratio change: 27.37%.
- Mean per-batch teacher/transition time: 9.6719/5.9458 ms; estimated direct 300-epoch overhead 487.33 s.
- Peak CUDA memory: 3840.62 MiB.

Machine-readable details are included in `formal_training_metadata.json` and `mtrd_training_summary.csv`.

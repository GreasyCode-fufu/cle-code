# MTRD vs Gram RelKD Gradient Diagnostic

- Gradient ratio status: WARNING
- Redundancy status: DISTINCT_GRADIENT_SIGNAL
- Scope: five real training batches; no optimizer step.

| Scale | Det grad | Weighted MTRD grad | MTRD/Det | Gram grad | MTRD-vs-Gram cosine |
|---|---:|---:|---:|---:|---:|
| P3 | 0.516414 | 0.0667065 | 0.136007 | 0.0628859 | 0.269741 |
| P4 | 0.367271 | 0.117072 | 0.293076 | 0.111338 | 0.269917 |
| P5 | 0.18331 | 0.119515 | 0.658108 | 0.216612 | 0.251434 |

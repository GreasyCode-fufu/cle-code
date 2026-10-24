# Round10: CLE token adaptation

18 neural trainings, full original train2386/val538. No test, grades or relabeling.
Exploratory checkpoint-selected development results; conditional intervals do not remove repeated selection bias.

| Arm | Mean val F1 | Malignant recall | Patient equal accuracy | Trainable |
|---|---:|---:|---:|---:|
| frozen_head | 0.865834 | 0.720085 | 0.882986 | 2307 |
| full_finetune | 0.892762 | 0.788462 | 0.890656 | 21631107 |
| lora_qv | 0.872516 | 0.747863 | 0.863466 | 149763 |
| local_adapter | 0.875194 | 0.726496 | 0.882101 | 153411 |
| cle_cls_context | 0.875779 | 0.713675 | 0.876636 | 152547 |
| cle_mean_context | 0.872422 | 0.732906 | 0.877760 | 152547 |

Complete preregistered structure gate: False.
Five paired comparisons and family intervals in paired_comparisons.json. Independent audit and patient confirmation required.
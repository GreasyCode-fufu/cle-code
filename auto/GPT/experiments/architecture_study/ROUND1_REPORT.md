# CLE architecture round 1: development results

Exploratory, validation-selected scores. The intervals condition on the fitted pairs and do not correct for model selection or multiple candidates.
No test images were evaluated by this architecture study.

| Recipe | Variant | Mean val F1 | Delta to paired plain ResNet | Patient interval | Extra parameters |
|---|---|---:|---:|---|---:|
| ac_only | conv_control | 0.873465 | +0.001052 | [-0.012397122556400358, 0.015320817124239088] | 46592 |
| ac_only | ls_reference | 0.875350 | +0.002937 | [-0.016251064685347344, 0.02252050883856918] | 94432 |
| ac_only | cle_context | 0.868542 | -0.003870 | [-0.021950567651696017, 0.016314488661959305] | 44808 |
| grade_weighted | conv_control | 0.874385 | -0.007573 | [-0.031855349462419434, 0.012622947214746837] | 46592 |
| grade_weighted | ls_reference | 0.881937 | -0.000021 | [-0.0204315511713046, 0.016912271305558044] | 94432 |
| grade_weighted | cle_context | 0.873220 | -0.008738 | [-0.022820696757301213, 0.003928973281763883] | 44808 |

Status: requires mechanism analysis and an independently specified confirmation/ablation round, not a publication-success declaration.
The architecture_sources, checkpoint SHA256, full per-image predictions, branch off diagnostics and training logs are preserved.
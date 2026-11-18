# 实验台账

本表区分正式 test 结果、验证/历史记录和未统一复核的旧分支。不得把“源码仍在”解释为“最终模型包含该模块”。

| 路线 | 状态 | 关键结果/证据 | 结论 |
|---|---|---|---|
| 原始 YOLOv8n | 历史基线 | 本项目 test mAP50-95≈0.429；约 3,140,748 Params、8.3 GFLOPs | 仅作本项目基线 |
| L1 | 历史结构搜索 | 历史 YAML/训练目录保留；未在本次交接中重跑 test | P4 容量压缩探索 |
| L1-B | 历史结构搜索 | 结果和配置保留在原始 runs/YAML；本交接不补造缺失的统一 test 指标 | 分支分析 |
| L1-N | 历史结构搜索 | 结果和配置保留在原始 runs/YAML；本交接不补造缺失的统一 test 指标 | 分支分析 |
| L1-96 | Teacher/有效结构 | fused 2,641,337，GFLOPs≈7.2，test mAP50-95=0.429，mAP75=0.460 | 正式 Teacher |
| SR1 | 历史压缩搜索 | 相关 YAML/训练记录保留；P5 压缩促成后续 P4Only 定位 | 分析路线 |
| SR1-P4Only | Student 结构基线 | 历史 test mAP50-95≈0.427；fused 2,603,129，GFLOPs=7.044096 | 最终 Student 推理结构 |
| C2f_SPP | 被支配 | test mAP50-95≈0.420 | 不进入最终方案 |
| C2f_DSM | 被支配 | test mAP50-95≈0.422 | 不进入最终方案 |
| Paired NoKD | 严格对照 | test mAP50-95=0.42120，mAP75=0.43523 | Gram 的因果配对基线 |
| Gram RelKD | **正式主方法** | test mAP50-95=0.435680，mAP75=0.459626 | **FINAL MAIN METHOD** |
| MTRD | 有效但次优 | test mAP50-95=0.432186，moisture=0.325034 | 关系蒸馏消融，不替代 Gram |

## 夜间组合蒸馏探索（均为 300 epochs，E07 未启动）

| ID | 方法 | best val mAP50-95 | test mAP50-95 | 状态 |
|---|---|---:|---:|---|
| E01 | Gram+MTRD 75/25 | 0.422408 | 未选测 | 探索完成 |
| E02 | Gram+Soft Foreground 70/30 | 0.428234 | 未选测 | 探索完成 |
| E03 | Gram+Attention Transfer 80/20 | 0.422504 | 未选测 | 探索完成 |
| E04 | Gram+P5 Adversarial Feature KD 80/20 | 0.431527 | 0.420200 | 失败；低于 NoKD，GAN 无证明收益 |
| E05 | Gram+Decoupled Class/Background 60/25/15 | 0.428120 | 未选测 | 探索完成 |
| E06 | Gram+ResponseKD 70/30 | 0.434847 | 0.429919 | 最佳探索项，但低于 Gram 0.435680 |
| E07 | Gram+P4/P5 GAN 80/20 | — | — | 时间预算不足，未启动 |

E06 可作为未来经明确批准的多 seed 补充研究，但不能替代正式 Gram RelKD。E04 判别器长期单边，且 test 低于 Paired NoKD。夜间详细证据见 `runs/analysis/nightly_kd_sweep_20260801/final_report.md`。


交付清单:
- /home/login/data/cle/auto/Network/EF-ConvNeXt/ (423M)
- 结构: README/demo.py/models(Convnext+convnext_ep)/weights(best+5seeds)/eval(evaluate复现通过F1 0.9598)/jobs(6个复现清单)/paper_data(5份CSV)/reports(4份文档)
- 验证: demo单图推理✓ evaluate test复现✓ MACs 3.41G(-23.7%)
- 源仓库commit 9afe1eb, 交付时间 2026-11-02 17:20
交付副本2:
- /home/login/data/cle/auto/prueNetwork/EF-ConvNeXt (423M)
- 新增 INNOVATIONS_AND_RESULTS.md(创新点详解+五域完整指标)
- demo --cpu 验证通过

## TCGA 新数据集对照 (2026-11-03, 6类36k/5.7k, 3seeds, commit 5c0bc98)
- baseline convnext_tiny: F1 84.38±0.41 / Kappa 84.87 / MCC 84.99 / BalAcc 85.05
- CLE-Net (egb_scg_hsc):  F1 85.52±1.82 / Kappa 86.18 / MCC 86.28 / BalAcc 86.95
- Δ: F1 +1.14 / Kappa +1.31 / MCC +1.29 / BalAcc +1.90 — CLE-Net 全面胜出
- 注意: CLE-Net 方差大(±1.82, s44=83.39 弱于 baseline s43/44), 中位数 85.33 vs 84.61
## TCGA egb_scg 结果 (2026-11-03 03:04, CLE-Net审计后最终形态EGB+SCG无HSC)
- egb_scg: F1 84.02±1.68 (84.65/85.69/81.73) — 低于 hsc 版 85.52 与 baseline 同量级
- 结论: TCGA 上 HSC 版 > 无 HSC 版(与 CLE 主域审计结论相反, 域特异; CLE-Net 论文叙事
  仍按其审计以 egb_scg 为最终形态, TCGA 两版并报)
- 事故修复: 链式调度 pgrep 自匹配死锁(父shell heredoc 含匹配串), GPU 空转 50min;
  改纯顺序执行脚本, 03:1x Wave B 已接棒(v2bc×3)

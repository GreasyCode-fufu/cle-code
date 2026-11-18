# CUFIT Full 对比归档

主汇总：`../analysis_3seed_summary.md`（含CUFIT的Full主指标、补充总体指标、分类别指标、三个种子明细及表9完整方案对比）。

详细报告： [COMPARISON_REPORT.md](COMPARISON_REPORT.md)。

- SUMMARY.csv / PER_SEED.csv / PER_CLASS.csv / RESULTS.json：CUFIT、Ours与RVC/RC参考组的统计，数值为0–1尺度。
- CUFIT_FULL_SEED_METRICS.csv：额外补齐MacroSpec、AUC-PR、BalancedAcc等指标。
- predictions/：本次汇总用到的12份逐样本预测副本。
- PREDICTION_SOURCES.json：预测原路径、副本及SHA256。
- audit.json / execution_files_sha256.json / references.json / EXPERIMENT_PLAN.md：数据、代码、权重与方案来源。
- INTEGRATION_AUDIT.json：主总结更新记录，以及更新前的精确备份路径和校验值。
- integration_backups/：修改前4份总结文件；仅用于回溯，不代表当前结果。

训练权重与完整日志保存在 `/home/login/data/cle/sunfuxiang/CUFIT0613/experiment_full_vs_ours_20261022/`，没有重复复制大权重。

主结论：CUFIT Full Macro-F1 90.43±0.38%，HQ + Ours 91.05±1.00%，后者描述性均值高0.619个百分点；未做显著性检验。

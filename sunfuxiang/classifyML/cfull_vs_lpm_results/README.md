# C_Full 与 C_LPM 三方交集对比：已完成

2026-10-23 06:35 完成15个完整CUFIT折外任务及6次Ours训练；最终独立核验通过，服务器watchdog已正常退出。

- [完整对比报告](COMPARISON_REPORT.md)
- [实验方案](EXPERIMENT_PLAN.md)
- [三种子总体统计](DOWNSTREAM_SUMMARY.csv)
- [逐种子结果](DOWNSTREAM_PER_SEED.csv)
- [逐类别结果](DOWNSTREAM_PER_CLASS.csv)
- [筛选集合数量与类别组成](SUBSET_COUNTS.csv)
- [最终独立核验](FINAL_VERIFICATION.json)

本轮配对重训结果：原LPM交集 Macro-F1 90.94±1.62%，完整CUFIT交集89.70±0.72%，后者均值低1.236个百分点；未作显著性检验。旧LPM使用整个外层训练折，而完整CUFIT留出内层选模患者；交集大小和类别组成也不同，不能把差值全部归因于adapter或样本质量。

历史LPM最高结果与本轮重训对照是不同实验，不能混用。项目主总结 `../analysis_3seed_summary.md` 已追加本轮配对表，保留历史记录。

完整权重、日志和分组清单保留在 `/home/login/data/cle/sunfuxiang/CUFIT0613/experiment_cfull_oof_20261023/`。

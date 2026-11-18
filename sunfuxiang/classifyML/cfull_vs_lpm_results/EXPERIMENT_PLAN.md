# 三方共识：C_LPM 与完整 CUFIT C_Full 的配对实验

日期：2026-10-23。主目录：`/home/login/data/cle/sunfuxiang/CUFIT0613/experiment_cfull_oof_20261023/`。

## 问题与预先固定的比较

检验把三方共识中的 C 从 DINOv2-LPM 改为完整 CUFIT 后，训练集合与下游分类性能如何变化。固定 R、V、样本ID、阈值、外层患者划分与最终 Ours 架构；不按测试表现改参数、挑种子或选择结果。

- 主臂 A：R∩V∩C_LPM，复用已核对的历史 2880 张样本集合。
- 主臂 B：R∩V∩C_Full，数量由新折外预测决定，不预设它更好。
- 两臂都新训练 Ours seed 42/43/44，共6个下游run，使用同一配置和环境。既有最高Ours结果只作历史参考，不替代本次配对对照。
- 主指标 Macro-F1，辅以 Accuracy、Macro-OvR-AUROC、恶性类 Recall/F1、各类指标和混淆矩阵；均值±样本标准差（ddof=1）。不宣称统计显著。

## 数据与泄漏控制

Full train=4977、val=538、test=671。审计从各split的processed CSV读取 leakage_group_id；三个split的样本ID、患者组都无交集（train 199组、val 39组、test 41组）。

外层：复用 `CUFIT0613/folds_3class_newnormal.json`，seed42/43/44各5折。外层留出患者不参加训练、早停或选模，每图恰好3次外层预测。

内层：在每个外层训练患者中使用 `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=20261023+seed*10+fold)`。按固定顺序取首个训练和验证都含3类的划分，约10%图像作内层验证。内外层分组清单在训练前写入 `folds/<job>/manifest.json`。独立538张val仅供后续Ours选模，不用于CUFIT折内训练；test不参与任何选型。

## 完整 CUFIT 选择器

15个折内完整训练，单GPU顺序执行。复用已验证训练代码快照，仅替换数据加载器为上述train/inner_val清单，并额外保存预热线性头；训练损失、更新规则、调度和早停不改。

- DINOv2 ViT-S/14固定骨干、双Rein、CLS特征；CE，无额外噪声，训练WeightedRandomSampler。
- batch32；Adam lr=1e-3、weight_decay=1e-5。
- 线性头预热5 epoch；课程阶段最多150 epoch；MultiStepLR在75/112/135衰减；内层验证Macro-F1选最佳LAM权重，早停50。
- 最佳LAM预测外层留出患者；三次正确预测数≥2即C_Full通过，再与固定R、V取交集。
- 同时用预热后LPM头对相同外层患者预测，作为内层训练子集匹配的选择器诊断；该诊断集合不额外训练下游模型。

## 配对 Ours 训练

两臂都使用当前冻结的 `train_cle_fair.py` 及 `resnet34_degconv_cpca`，不修改原项目源码。ImageNet ResNet34权重的SHA256必须为已审计值；所有兼容骨干键严格加载，允许随机初始化的新增DegConv/CPCA及分类头键列表已预先锁定为108项。

- seed42/43/44，batch32，224输入；未加权CE，label smoothing=0。
- AdamW lr=1e-4、weight_decay=.05，AMP，CosineAnnealingLR(T_max=300)，最多300 epoch，val Macro-F1早停30。
- 保留当前fair脚本的训练/验证变换；两臂测试均使用相同原Ours变换：短边Resize256、CenterCrop224、ImageNet归一化。
- 每个run只在训练完成后评估同一个671张test；保存高精度逐样本概率。

## 解释边界

这是两个筛选器协议的整体比较。旧C_LPM使用整份外层训练折，新C_Full需要从外层训练折中留出内层选模患者，因此阶段和训练数据量同时变化，不能将全部差异严格归因于adapter；同内层子集的LPM信号诊断可辅助解释。

交集大小和类别组成可能不同。本阶段没有等规模/等类别组成对照，不把“更好分类结果”等同于已证实的图像质量提升。既有测试集已在历史研究中查看，结果属于探索性比较；不按测试结果选择C版本或阈值。

## 运行、恢复和监控

- `runner.py` 持有文件锁，后台脱离SSH运行；每15秒更新心跳。
- 按折及下游run保存完成标记、输入/输出哈希；重启跳过已验证完成任务。未完成训练从固定种子重跑，旧目录移入当前任务的 interrupted 目录，不覆盖历史完整产物。每个任务最多自动尝试2次。
- 每次启动新GPU任务前检查是否有其他计算进程；若有则等待。
- 服务器watchdog.py每60秒检查心跳、进程和日志，在monitor_status.json与watchdog.log记录状态；runner意外退出且无活跃子任务时按原协议恢复，达到尝试上限则记录需人工处理。聊天定时监控创建被自动审批拒绝（需要审批但当前策略不允许），不能主动发送聊天通知。
- 最终自动生成报告，发布到 `classifyML/cfull_vs_lpm_results/`，并将配对比较追加到 `classifyML/analysis_3seed_summary.md`，保留其原内容和备份。

## 输出

`status.json`、`runner.log`、`logs/`、`folds/`；`results/C_FULL_OOF.csv`、`SUBSET_COUNTS.csv`、`SAMPLE_CHANGES.csv`、`SELECTION_COMPARISON.json`、`DOWNSTREAM_SUMMARY.csv`、`DOWNSTREAM_PER_SEED.csv`、`DOWNSTREAM_PER_CLASS.csv`、`COMPARISON_REPORT.md`、`COMPARISON_RESULTS.json`。

代码、数据、分组、阈值和来源见 `audit.json`、`EXECUTION_MANIFEST.json`、`outer_folds.json`、`manifests/`、`pretrained_load_contract.json`。正常阶段不根据测试结果调整本方案。

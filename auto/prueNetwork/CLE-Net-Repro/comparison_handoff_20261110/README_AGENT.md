# 当前使用说明（优先于后文历史归档说明）
当前Excel为experimental_results_3seed_comparators.xlsx：保留原8页，新加9—11页仅含ZCode的Jumbo-small/GmNet-s3/iFormer-S/LSNet-S，42/43/44三种子。未新增ZCode CLE-Net或baseline。CLE-Net由用户自行整理。
当前有效机器可读汇总：r6b_3seeds_four_networks.json、r6b_3seeds_four_networks_summary.csv、r6b_3seeds_four_networks_per_seed.csv。其余5种子文件与原始对照仅作历史溯源，不用于覆盖当前主表。
统计由原始完整精度JSON重算，并与ZCode提供的3种子表逐行核验Macro-F1。原Excel前8页单元格均未改变。
README_AGENT.md、AGENTS.md为后续Agent入口。没有重新训练任何网络。

# 对比实验交接包 / Agent 入口

## 首先阅读
本目录为2026-11-10创建的独立快照。ZCode源文件均只读复制，禁止修改原路径。重新实验请另建工作目录；不得在本快照中覆盖结果。
SOURCE_INDEX.csv列出每个来源与快照位置。FILE_MANIFEST.csv逐文件记录源/副本SHA256，所有副本已与源文件校验。

## 实验分组与归属
- zcode_r6b：ZCode独立完成的Jumbo-small、GmNet-s3、iFormer-S、LSNet-S，以及其5种子CLE-Net/ConvNeXt-T对照，种子42—46。环境sunfuxiang，torch2.12.0+cu130。不是Codex运行结果。
- codex_topnets：Codex五网络（MambaOut、MambaVision、TransNeXt、ThinkingViT、CPUBone）及本Agent独立复跑的B环境对照，种子42/43/44，cle_topnets、torch2.5.1+cu124。禁止将A参考CLE-Net的0.9630替换到这个批次。
- codex_reference_A：已核查历史参考条件下的独立复跑，CLE-Net 0.9630±0.0053，baseline 0.9587±0.0054，3种子。即使与ZCode使用同名环境，也不自动合并批次或种子。
- previous_paper_data：历史主干、消融、各数据集及容量实验的已审查CSV，文件自带协议标记。部分为历史ZCode来源，不得统一归称Codex新跑。
- stopped_mobile_transfer：MobileNet-V3迁移实验已被用户明确停止。保留中断权重和部分结果，不得自动恢复、补跑或作为完整3种子实验引用。

## 结果使用
优先使用每次运行eval_*/results.json（按test_dir末级test且n_total=576确认测试集，不能仅凭目录名推断），而不是交接消息中的四舍五入数字。r6b_per_seed_recomputed.csv保留9指标；r6b_summary_recomputed.csv由原始精度重算均值及样本标准差ddof=1。全部采用0—1尺度。
原始r6b_summary.json采用百分数且可能与消息中“3种子”说明混淆，本目录重算表为5种子。交接消息中的逐种子数字与所列均值不自洽，切勿照抄。
主指标是test Macro-F1。训练仅按val Macro-F1选checkpoint。n_test=576。数据集路径继承原始run_meta；不在此复制原始患者图像。
ZCode与Codex B环境不同，务必分别成表，使用本批次自己的对照。不得混合不同环境或不同训练批次计算SD。

## 程序位置与运行条件
ZCode适配器：zcode_r6b/training_code/models/sota2026.py。官方实现及精确commit见official_repositories中的MANIFEST.tsv。GmNet须保留f12_bn=True、dwconv2_bn=True，LSNet动态Triton核只能在GPU使用。
ZCode纯张量预训练权重在zcode_r6b/pretrained。禁止将原始npy上传件当作通用torch checkpoint加载。
Codex新网络入口：codex_topnets/latest_top_networks/run_train.py、run_eval.py、adapters.py；其父训练代码保存在parent_training_code。原代码含历史绝对路径和包导入假设，快照不声称直接搬目录即可运行。重新运行前在新工作副本中恢复父目录布局，依据原run_meta和启动日志核对命令及权重路径。
A参考运行入口和历史冻结代码位于codex_reference_A/historical_protocol/code及其运行脚本；锁定版本、数据manifest和初始化条件见repro_metadata。以项目根README和protocol_lock.json为准。
共同协议：224输入，ImageNet归一化，AdamW lr1e-4、wd0.05，batch32，cosine，最多300轮、early-stop30。EGB门控lr0.01、wd0。实际细节以对应原始run_meta/日志/源码为准，不能仅根据这段概述推断完全相同。

## 复杂度与限制
zcode_r6b/reports/r6_bench_latency.json是ZCode独立测量，未与Codex硬件/版本口径强行合并。LSNet CPU=NA。MACs/FLOPs的自动统计存在FFT、attention、selective scan、Triton等漏计，不能宣称完整算子复杂度；FLOPs估计如使用2×MACs须明示。
临床患者独立性未验证；既有固定测试集曾参与开发反馈，不得将这批数据解释为严格前瞻性临床验证。

## 三种子交接已完成
已按用户更新版交接完成原始记录核验。

## 最新用户要求（覆盖上面的待办）
已收到并核验ZCode三种子更新。仅将四个对比网络（42/43/44）纳入新表，不新增ZCode CLE-Net或baseline结果，CLE-Net由用户自行整理。当前有效文件为r6b_3seeds_four_networks.json及两个同名前缀CSV。更新版原始报告在zcode_r6b/update_3seeds，保留原文件中对照信息用于溯源，不表示本次Excel采用这些对照。旧五种子归档不删除、不混用。

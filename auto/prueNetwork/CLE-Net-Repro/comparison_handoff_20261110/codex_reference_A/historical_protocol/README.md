# CLE_ac 历史协议独立重跑

执行时间：2026-11-09 17:47—18:10（Asia/Shanghai）。由本轮 Codex 独立启动，共六次从 ImageNet 预训练开始的训练；未复用历史训练检查点，也未采用 Zcode 的重跑结果。全部训练、验证评估及测试评估完成后，才读取历史检查点作复现核对。

## 结果

| 模型 | Seed 42 Macro-F1 | Seed 43 Macro-F1 | Seed 44 Macro-F1 | 均值±样本标准差 |
|---|---:|---:|---:|---:|
| ConvNeXt-T baseline | 0.963587 | 0.959687 | 0.952851 | 0.9587±0.0054 |
| CLE-Net（EGB+SCG） | 0.961691 | 0.958511 | 0.968832 | 0.9630±0.0053 |

CLE-Net 相对本次 baseline 的均值差为 **+0.430323 个百分点**。逐种子差为 −0.189541、−0.117613、+1.598124 个百分点，三个种子中一次为正。均值优势不能表述为每个种子都优胜或统计显著优势；本次没有做显著性检验。

| 指标 | ConvNeXt-T baseline | CLE-Net |
|---|---:|---:|
| Accuracy | 0.9589±0.0053 | 0.9630±0.0053 |
| Weighted-F1 | 0.9589±0.0053 | 0.9630±0.0053 |
| Balanced Accuracy | 0.9590±0.0058 | 0.9629±0.0050 |
| Macro-AUROC | 0.9961±0.0006 | 0.9959±0.0004 |
| Macro-Sensitivity | 0.9590±0.0058 | 0.9629±0.0050 |
| Macro-Specificity | 0.9795±0.0027 | 0.9814±0.0026 |
| Cohen’s κ | 0.9382±0.0080 | 0.9443±0.0080 |
| MCC | 0.9384±0.0080 | 0.9444±0.0080 |

每个种子均在固定576张测试图像上独立评估，再以原始精度计算均值和样本标准差（ddof=1）。Balanced Accuracy 与 Macro-Sensitivity 在本任务定义下相等。未集成不同种子的预测。

## 复现核对

对 CLE-Net，参考为原项目 `weights_cle_fixed_v1` 的 seed42/43/44：

- 三个种子的九项测试指标差值全部为0。
- 三个种子的训练CSV文本完全相同。
- 每个种子的293个 state_dict 张量逐项 `torch.equal` 为真，最大参数绝对差为0。

对 baseline，参考为原项目10月30日的 `weights_cleac_baseline_v2`：

- 三个种子的九项测试指标差值全部为0。
- 三个种子的训练CSV文本完全相同。
- 每个种子的182个 state_dict 张量逐项相同，最大参数绝对差为0。

以上结论比较的是张量内容，不要求两个 PyTorch 序列化文件的二进制文件哈希相同。详情见 `reproducibility_audit.json`。

这次说明，在本次恢复的历史源码、环境和设置下，历史 CLE-Net 结果能够完整复现。此前在另一套代码和环境下得到0.9510的结果，不能直接作为“同条件重跑每次不同”的证据。代码版本与环境均曾变化，本次没有通过单因素实验进一步区分二者的影响。

早期论文表中的 baseline 0.9489±0.0122 并非本次重跑数值，因此不能把旧的 +1.41pp 继续写成本次同批对比增益。另：10月30日旧报告写过 baseline 0.9587±0.0045；对其三个原始测试浮点值重新计算，样本标准差实际为0.0054343458。本报告统一使用0.0054，不沿用旧报告标准差。

## 运行条件

- 原项目：`/home/login/data/cle/auto/Network/CLE-Net`。
- 本次目录：`/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109`。
- 历史源码：通过 Git archive 冻结 `b457d93e13fca274876cfd7aa040d7db3a5aef15`，独立放置于本次目录的 `code/`；原项目源码未修改。
- 执行环境：`/home/login/miniconda3/envs/sunfuxiang/bin/python`；Python 3.12、PyTorch 2.12.0+cu130、Torchvision 0.27.0+cu130。
- 数据：CLE_ac_v1，2077/427/576，三类；逐文件SHA256保存在 `dataset_manifest.json`。患者独立性仍未由本次训练验证。
- 初始化：`convnext_tiny_1k_224_ema.pth`；SHA256 为 `14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595`，六个运行均核对一致。
- 输入224×224；ImageNet mean/std；增强与验证、测试预处理沿用冻结源码的原始实现。
- AdamW，普通参数 lr=1e-4、wd=0.05；CLE-Net的15个门控参数 lr=0.01、wd=0；未加权CE，label smoothing=0；AMP。
- batch32，最多300轮，CosineAnnealingLR；验证Macro-F1连续30轮未提升早停；仅按验证集选择检查点。
- DataLoader workers=3；普通shuffle；OMP/MKL线程各2。
- 同一seed先baseline、后CLE-Net，六个任务依次执行；每次训练启动前检查其他GPU计算任务，存在时等待，不终止其他任务。

## 文件

- `summary.csv` / `summary.json`：九项指标的三种子均值和样本标准差。
- `per_seed.csv`：六次测试的完整指标、最佳epoch、停止epoch与原始JSON路径。
- `paired_differences.json`：逐种子CLE-Net减baseline的Macro-F1差值。
- `protocol.json`：源码commit、预训练与源码归档哈希、超参数。
- `environment_versions.txt` / `pip_freeze.txt`：本次依赖版本。
- `events.jsonl`：实际启动时间、PID、训练及评估命令、退出码。
- `reproducibility_audit.json`：与历史指标、训练日志及权重张量的核对。
- `dataset_manifest.json`：划分内图像相对路径和内容SHA256。

服务器还保存全部六次训练的检查点、CSV日志、run_meta、验证/测试的results.json及predictions.csv，以及源码归档 `historical_source.tar`。下载结果包不含大型模型权重。

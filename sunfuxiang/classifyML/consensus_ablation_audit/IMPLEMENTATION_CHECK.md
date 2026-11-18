# CLE 共识消融实现检查

检查日期：2026-09-16  
项目：`/home/login/data/cle/sunfuxiang/classifyML`  
Python：`/home/login/miniconda3/envs/sunfuxiang/bin/python`  
边界：未启动 epoch training，未调用 GPU，未生成 Random-2880。

## 1. 已实现文件

| 文件 | 作用 |
|---|---|
| `train_cle_fair.py` | 新增唯一 CLI 参数 `--train-manifest`；不传时走历史 Full 扫描；传入时严格过滤 |
| `consensus_manifest.py` | manifest 解析、路径/ID/类别/泄漏检查、计数和 SHA-256 |
| `scripts/build_consensus_ablation_manifests.py` | 生成并验证 R/V/C/RV/RC/VC；只重算 RVC，不生成 RVC 训练 manifest |
| `scripts/validate_consensus_ablation_data.py` | CPU-only Full 回归和六组单 batch 验证 |
| `scripts/summarize_consensus_ablation.py` | 从 predictions.csv 统一重算指标并按三 seed 用 ddof=1 汇总 |
| `tests/test_consensus_ablation.py` | manifest、泄漏防护、run_meta、指标和样本标准差测试 |

原始训练脚本快照保存在：

`consensus_ablation_audit/preimplementation_snapshot/train_cle_fair.py.before_manifest`

其 SHA-256 为 `0c3a690efa4a2bf36b57df68e203b9835e6005da169e5473f3232020d8ad3a16`。

## 2. TDD 与静态检查

- RED：测试首先因 `consensus_manifest` 不存在而失败；run_meta 测试首先因 `build_run_meta` 不存在而失败。
- GREEN：最终标准库 unittest 共 8 项，全部通过。
- 覆盖行为：有效筛选；重复 sample_id；越界路径；类别/父目录不一致；val/test sample_id 泄漏；无 manifest 旧 run_meta schema；manifest run_meta 完整字段；统一指标和 ddof=1。
- 所有新增/修改 Python 文件均通过 `py_compile`。
- 服务器环境没有 pytest，因此没有安装或改动环境，使用标准库 unittest 完成等价的快速测试。

## 3. 无 manifest 向后兼容

修改前 Full 扫描基线：

```text
count = 4977
Normal = 1578
Benign = 2136
Malignant = 1263
ordered path+label SHA-256 = 2a365f3576326c3a40dadf9c0ab2d4f5ca4b2a8b42875445c56adeb75647f021
```

修改后由 `train_cle_fair.prepare_training_samples(..., train_manifest="")` 直接读取所得 count、三类计数和有序路径/标签哈希与上述基线完全相同。无 manifest 时 run_meta 仍保持旧 13 字段 schema，不加入新字段。

## 4. manifest 生成与校验

生成目录：`manifests/consensus_ablation/`。

生成文件：R.csv、V.csv、C.csv、RV.csv、RC.csv、VC.csv、MANIFEST_METADATA.json。没有生成 RVC.csv。

所有 manifest 均通过：sample_id 唯一、relative_path 唯一、真实文件存在并位于 Full train、CSV 记录路径与当前文件 basename/类别一致、manifest 类别与父目录一致、无未知 ID、与 val/test 无 sample_id 重叠、实际筛选数量等于 manifest 行数。详细结果见 `MANIFEST_VALIDATION.csv`。

RVC 三方复核：

```text
computed RVC = 2880
authoritative consensus CSV = 2880
existing HQ train = 2880
computed ↔ authoritative 双向差集 = 0 / 0
computed ↔ HQ 双向差集 = 0 / 0
```

## 5. CPU/dry-run 结果

- 强制 `CUDA_VISIBLE_DEVICES=''`。
- 六个 subset 均只构建 Dataset/DataLoader 并读取一个 batch。
- 每组 `images.shape=[32,3,224,224]`，`labels.shape=[32]`，labels dtype=`torch.int64`。
- `training_started=false`，未构建 epoch 循环。
- fixed val=538，fixed test=671。
- ResNet34 ImageNet 权重 SHA-256 再次验证为 `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef`。
- CPU 构建 3-class ResNet34 并按正式逻辑加载预训练权重成功；仅缺预期的 `fc.weight`、`fc.bias`，unexpected keys 为空。

## 6. run_meta 增强

增强字段只在 manifest run 中加入，包含：subset、manifest path/hash、实际训练总数与三类计数、batch/max epochs、AdamW、lr、weight decay、CosineAnnealingLR、label smoothing、early stop、AMP、选优指标、预训练路径/hash、train/val root、git commit、代码哈希和带时区 timestamp。

当前目录没有可获取的 Git commit，因此未来 manifest run 的 `git_commit` 为 null；`code_hash` 会记录实际训练时的 `train_cle_fair.py` 和 `consensus_manifest.py` SHA-256，满足可追溯要求。

## 7. 旧预测锁定与统一汇总

`LEGACY_PREDICTION_MAP.json` 显式锁定 Full/RVC × seeds 42/43/44 的六个 exact predictions.csv，并记录各自 SHA-256 与 671 行数。

统一汇总 dry-run 已：

- 校验六个文件哈希；
- 读取每个 671 行预测；
- 从 true/pred/probability 统一重算 Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1、Macro AUROC OvR、每类 precision/recall/F1 和混淆矩阵；
- 按 ddof=1 汇总三 seed；
- 得到 Full Macro-F1 `0.876360 ± 0.002346`，RVC Macro-F1 `0.901115 ± 0.009829`；
- 正确把尚未训练的 R/V/C/RV/RC/VC 共 18 个 predictions 标记为预期缺失。

正常（非 dry-run）模式会在任何一个预期 predictions 缺失时拒绝输出最终八组汇总。

## 8. 安全边界检查

- `weights_consensus_ablation/` 尚不存在，因此没有正式训练产物。
- 六个旧 Full/HQ 权重目录在本日新增或修改的文件数为 0。
- val/test 未修改；未复制任何原始图像。
- 所有新增文件属主为 `login:login`。

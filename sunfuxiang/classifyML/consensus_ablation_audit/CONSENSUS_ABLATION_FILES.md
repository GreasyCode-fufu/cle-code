# 共识消融文件审计与拟新增文件

## 1. 已核验的输入文件

| 文件 | 用途 | 审计结果 |
|---|---|---|
| `outputs_3class_resnet_newnormal/train_sample_grading.csv` | R-good | 4977 行唯一 sample_id；A/B/C 得 3500 |
| `outputs_3class_vit_newnormal/train_sample_grading.csv` | V-good | 4977 行唯一 sample_id；A/B/C 得 3474 |
| `../CUFIT0613/cufit_lpm_oof_3class_newnormal.csv` | C-good | 4977 行唯一 sample_id；correct_rate≥0.5 得 3749 |
| `../CUFIT0613/consensus3_oof_good_sample_ids.csv` | 权威 RVC | 2880 行唯一 sample_id；与重算 RVC 双向差集 0 |
| `datasets/3class/train` | Full train | 当前真实位置在上级项目目录；4977 张 |
| `datasets/3class_highquality/train` | 既有 RVC/HQ train | 2880 张；与权威 RVC 双向差集 0 |
| `datasets/3class/val` | 固定 val | 538 张 |
| `datasets/3class/test` | 固定 test | 671 张 |

上述相对路径中的 datasets 实际根目录为 `/home/login/data/cle/sunfuxiang/datasets`，不是当前已失效的 `/home/login/cle/sunfuxiang/datasets`。

## 2. 已核验的训练与评估代码

| 文件 | 结论 |
|---|---|
| `train_all_pretrained.sh` | 历史 Full/HQ 批量入口；超参数和 seeds 已核对，但硬编码旧路径，当前不可直接使用 |
| `train_cle_fair.py` | 实际单次训练入口；当前只接受实体类别目录，不支持 manifest |
| `cle_fair_common.py` | 模型注册和验证变换；三分类训练使用其中的模型构建/val transform |
| `run_eval_all.py` | 历史三分类批量评估/汇总入口；调用 `evaluate.py` |
| `evaluate.py` | 历史三分类实际评估入口；argmax、Macro-AUROC OvR |
| `evaluate_cle_fair.py` | 二分类阈值优化评估，不是本三分类消融入口 |
| `utils.py` / `my_dataset.py` | 目录扫描后构造路径/标签列表，证明可低风险加入 manifest 过滤 |

## 3. 可直接复用的旧结果

以下目录中的 ResNet34 权重、日志、`run_meta.json` 和 671 行 predictions 均存在：

```text
weights_3class_full_seed42/resnet34
weights_3class_full_seed43/resnet34
weights_3class_full_seed44/resnet34
weights_3class_hq_seed42/resnet34
weights_3class_hq_seed43/resnet34
weights_3class_hq_seed44/resnet34
```

`summary_full.csv`、`summary_hq.csv`、`summary_compare.csv` 可作为历史核对证据，但新八组对比应由统一汇总脚本从 predictions 重算，不能直接拼接旧 summary。

## 4. 本阶段创建的审计交付物

```text
consensus_ablation_audit/
├── CONSENSUS_ABLATION_PLAN.md
├── CONSENSUS_ABLATION_SUBSET_COUNTS.csv
├── CONSENSUS_ABLATION_COMMANDS.md
└── CONSENSUS_ABLATION_FILES.md
```

本阶段没有创建 manifest、数据软链接、权重或新评估结果。

## 5. 下一阶段拟新增文件（需先代码审查）

```text
classifyML/
├── consensus_manifest.py
├── manifests/consensus_ablation/
│   ├── R.csv
│   ├── V.csv
│   ├── C.csv
│   ├── RV.csv
│   ├── RC.csv
│   ├── VC.csv
│   └── MANIFEST_METADATA.json
├── scripts/
│   ├── build_consensus_ablation_manifests.py
│   ├── validate_consensus_ablation_data.py
│   ├── summarize_consensus_ablation.py
│   └── build_random2880_manifests.py
├── tests/
│   └── test_consensus_ablation.py
└── weights_consensus_ablation/
    ├── R/seed{42,43,44}/
    ├── V/seed{42,43,44}/
    ├── C/seed{42,43,44}/
    ├── RV/seed{42,43,44}/
    ├── RC/seed{42,43,44}/
    └── VC/seed{42,43,44}/
```

不为 RVC 创建新的训练定义或权重目录；RVC 始终指向已经核验的 HQ 2880 和旧三种子结果。

## 6. manifest 生成器的强制契约

生成器必须：

1. 以 sample_id 集合运算，不以行号或文件顺序运算；
2. 拒绝重复 ID、缺失 ID、未知 grade、NaN correct_rate；
3. 将旧 CSV 中的历史绝对路径改写为相对 Full-train 路径；
4. 核对父目录类别和 sample_id 对应关系；
5. 验证 R/V/C 输入 ID 宇宙完全相同；
6. 验证计算 RVC 与权威 RVC、现有 HQ 三方双向差集均为 0；
7. 输出每个 manifest 的总数、三类计数和 SHA-256；
8. 证明所有训练路径均在 Full train，且不与固定 val/test 相交；
9. 使用原子写入，已存在且哈希不同则拒绝覆盖。

## 7. 训练入口拟修改点

只新增可选 `--train-manifest`：先按现有代码扫描 Full train、建立 `sample_id/relative_path → (path,label)` 映射，再按 manifest 精确取子集。未传参数时行为必须逐字节兼容旧流程。新 `run_meta.json` 还应记录 batch size、max epochs、optimizer、lr、weight decay、scheduler、early-stop、label smoothing、预训练文件及哈希、manifest 路径及哈希、实际三类计数、代码版本和当前绝对数据路径。

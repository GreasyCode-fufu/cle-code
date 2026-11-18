# CLE 样本筛选共识消融实验计划

审计日期：2026-09-16  
项目：`/home/login/data/cle/sunfuxiang/classifyML`  
状态：仅完成只读审计与方案设计；未启动训练或评估，未修改旧权重和结果。

## 1. 科学问题与实验假设

现有结果只证明三模型共识集 RVC/HQ（2880）优于 Full（4977），尚不能区分收益来自：

1. 某一个筛选器本身；
2. 任意两个筛选器的共识；
3. 三个异构筛选器的增量共识；
4. 更小训练集带来的正则化效应；
5. 各方法改变类别组成带来的混杂。

第一轮固定分类器为 ResNet34，只改变训练样本集合，以 Macro-F1 为主指标。主消融比较 Full、R、V、C、RV、RC、VC、RVC；另设 Random-2880 尺寸与类别组成匹配对照。

## 2. 已审计的真实 Full/HQ ResNet34 协议

历史批量入口是 `train_all_pretrained.sh`，单次实际训练入口是 `train_cle_fair.py`。六个 ResNet34 `run_meta.json`、训练日志和当前代码相互吻合。

| 项目 | 已确认设置 |
|---|---|
| 模型 | `resnet34`，3 类 |
| seeds | 42、43、44 |
| 输入尺寸 | 224 |
| batch size | 32 |
| 最大 epochs | 300 |
| optimizer | AdamW |
| 初始 learning rate | 0.0001 |
| weight decay | 0.05 |
| scheduler | `CosineAnnealingLR(T_max=300, eta_min=0)`，每 epoch step |
| loss | 未加权 CrossEntropy；label smoothing=0 |
| AMP | CUDA GradScaler/autocast 开启 |
| early stopping | 验证 Macro-F1 连续 30 个 epoch 严格不提升即停止 |
| best checkpoint | 验证 Macro-F1 严格提升时保存 |
| train augmentation | Resize(256×256) → RandomHorizontalFlip(0.5) → RandomVerticalFlip(0.5) → RandomRotation(±15°) → RandomCrop(224×224) → ToTensor → Normalize |
| validation transform | Resize(256×256) → CenterCrop(224) → ToTensor → Normalize |
| normalization | ImageNet mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]` |
| initialization | PyTorch 官方 ImageNet ResNet34：`pth/resnet34-333f7ec4.pth`；SHA-256 `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef` |
| validation | 固定 538 张；Full/HQ 目录逐文件内容相同 |
| test | 固定 671 张；Full/HQ 目录逐文件内容相同 |
| test evaluation | `run_eval_all.py` 调用三分类 `evaluate.py`；argmax；Macro-AUROC 为 OvR macro |

六个历史训练实际运行 epoch 数为 Full 35/43/37、HQ 56/69/49；各日志的停止 epoch 均是最佳 epoch 后第 30 个未提升 epoch，证明早停行为与代码一致。

注意：二分类用的 `evaluate_cle_fair.py` 不是现有三分类 Full/HQ 汇总的实际评估入口。

## 3. 集合重算结果

从以下真实输入以唯一 `sample_id` 做集合运算：

- R：`outputs_3class_resnet_newnormal/train_sample_grading.csv`，`grade ∈ {A,B,C}`；
- V：`outputs_3class_vit_newnormal/train_sample_grading.csv`，`grade ∈ {A,B,C}`；
- C：`../CUFIT0613/cufit_lpm_oof_3class_newnormal.csv`，`cufit_lpm_correct_rate >= 0.5`；
- 权威 RVC：`../CUFIT0613/consensus3_oof_good_sample_ids.csv`。

四份输入均无重复 `sample_id`；R/V/C 三份输入覆盖相同的 4977 个 ID。重算得到：R=3500、V=3474、C=3749、RV=3103、RC=3141、VC=3115、RVC=2880。计算所得 RVC 与权威 RVC 文件双向差集均为 0；与当前 `datasets/3class_highquality/train` 的文件集合双向差集也均为 0。

类别组成详见 `CONSENSUS_ABLATION_SUBSET_COUNTS.csv`。RVC 相比 Full 明显提高 Normal 比例并降低 Malignant 比例，因此所有结果都必须同时报告 Macro-F1、每类指标和混淆矩阵，不能只报告 Accuracy。

## 4. 实际训练矩阵与复用

| Dataset | seed42 | seed43 | seed44 | 本轮动作 |
|---|---:|---:|---:|---|
| Full | 复用 | 复用 | 复用 | 旧 ResNet34 结果齐全 |
| R | 新训 | 新训 | 新训 | 3 runs |
| V | 新训 | 新训 | 新训 | 3 runs |
| C | 新训 | 新训 | 新训 | 3 runs |
| RV | 新训 | 新训 | 新训 | 3 runs |
| RC | 新训 | 新训 | 新训 | 3 runs |
| VC | 新训 | 新训 | 新训 | 3 runs |
| RVC | 复用 | 复用 | 复用 | 即现有 HQ，定义与文件均已核同 |

主消融新增训练量为 **6 subsets × 3 seeds = 18 runs**。Full 与 RVC/HQ 各三种子无需重训。

复用旧结果时，应固定使用每个运行目录中最新且字段齐全的 `eval_*/predictions.csv`，由统一汇总脚本从预测重新计算所有指标。这样可补齐旧 `results.json` 没有持久化的 Macro Precision 和 Macro Recall，同时避免新旧汇总逻辑差异。

## 5. 数据实现方案：推荐 manifest 过滤

当前 `train_cle_fair.py` 通过 `read_split_train_data(root)` 扫描 `root/<class>/*`，本身没有 manifest 参数；但它最终把路径和标签列表传给 `MyDataSet`，因此加入过滤的改动很小。

推荐方案 A：为训练入口增加可选的 `--train-manifest`，在完成 Full train 扫描后，按规范化后的 `sample_id`/相对路径过滤 `train_images_path` 与标签。旧调用不传该参数时行为必须完全不变。manifest 建议字段：

```text
sample_id,class_name,relative_path
```

必须增加的防错检查：ID 唯一、路径唯一、全部位于 Full train、标签与父目录一致、数量与预注册计数一致、无 val/test ID、manifest SHA-256 和最终三类计数写入 `run_meta.json`。

方案 A 优点：不复制图像、不制造数万个链接、路径迁移更容易、集合定义可审计；缺点：需对已验证训练入口做一个小的向后兼容改动，因此实施后必须做 CPU/dry-run 数据加载验证。

方案 B：构建 `datasets/ablation/<subset>/train/<class>/` 的相对软链接。优点是完全不改训练代码；缺点是链接数量多、容易悬空、迁移脆弱，且集合来源不如 manifest 清晰。仅在不允许修改训练入口时采用。禁止复制原图。

RVC 不重新定义、不重新生成训练集，直接复用现有 `datasets/3class_highquality/train` 和历史 HQ 结果。

## 6. 公平性与评价

所有新运行必须锁定：

- 唯一变化是训练 manifest；
- val 始终为当前 `datasets/3class/val` 的 538 张；
- test 始终为当前 `datasets/3class/test` 的 671 张；
- ResNet34、预训练权重及其 SHA-256、超参数、增强、归一化、早停和选优规则均与旧实验一致；
- 评估使用三分类 `evaluate.py` 的现有 argmax 协议和相同 test；
- 输出写入 `weights_consensus_ablation/<subset>/seed<seed>/`，绝不写入旧目录。

至少统一输出 Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1、Macro-AUROC (OvR)，另保留每类 precision/recall/F1/AUROC、混淆矩阵。三 seed 用样本标准差（ddof=1）报告 mean ± std，主指标为 Macro-F1。

## 7. Random-2880 对照设计

推荐建立 **5 个独立的分层随机训练子集**，sampling seeds 预注册为 202601–202605。每个子集从 Full train 内无放回抽样，并精确匹配 RVC 的类别计数：Normal=1303、Benign=1114、Malignant=463，总计 2880。

每个随机子集都用训练 seeds 42/43/44，各训练一次，形成 5 × 3 = 15 个额外 runs。不要只让一个抽样 seed 对应一个训练 seed，否则样本抽样波动会与优化随机性混杂。主报告应先对每个 random subset 的三个训练 seed 求均值，再报告五个 subset 均值的分布；也可补充 15 runs 的层级 bootstrap。RVC 的三个 seed 结果与 Random-2880 分布比较，用于判断收益是否超过“仅减少数据量并改变类别组成”的效果。

随机 manifest 必须保存 sampling seed、生成脚本版本、Full 样本清单 SHA-256、类别目标、选中 ID 和 manifest SHA-256；val/test 仍固定不变。该对照本阶段只设计，不生成、不训练。

## 8. 训练前必须解决的风险

1. **路径迁移**：历史脚本和 `run_meta` 记录 `/home/login/cle/...`，该路径当前不存在；当前真实根目录是 `/home/login/data/cle/...`。未来命令必须使用当前路径，不能直接运行旧 `train_all_pretrained.sh`。
2. **manifest 功能尚不存在**：实施向后兼容参数并通过计数、类别映射、首批数据加载和旧调用回归检查后，才能训练。
3. **旧元数据不完整**：旧 `run_meta` 未记录 batch size、epochs、lr、weight decay、scheduler、early-stop 和预训练源；新运行应全部写入，且保存代码/manifest/权重哈希。
4. **指标持久化缺口**：当前 `evaluate.py` 的 `results.json` 未保存 Macro Precision/Recall；统一汇总必须从预测表重算，或先向后兼容地补齐输出字段。
5. **评估目录有多个时间戳版本**：必须由 manifest 锁定每个复用运行所选的预测文件，不能依赖不透明的“最新目录”。
6. **多人共用服务器**：训练前检查 GPU 显存、负载、同账号进程、输出目录和端口；每个 run 使用独立目录并保留完成标记。端口转发 12789 的占用不影响本次审计，但说明共享端口确有冲突可能。
7. **统计解释**：筛选显著改变类别比例，且八个训练集是嵌套/重叠集合；报告效应量和每类结果，不把三个训练 seed 当作独立临床样本进行夸大的显著性推断。

## 9. 审计结论

主消融设计成立，新增训练量为 18 runs；固定 val=538、test=671 可以保证。Full 与 RVC/HQ 的旧三种子 ResNet34 结果可以直接复用，但汇总时应从已存 predictions 统一重算指标。开始训练前，首要工作是实现并验证 manifest 过滤、消除旧硬编码路径、扩充运行元数据与统一指标汇总。

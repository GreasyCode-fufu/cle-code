# bow_fusion — train_cle_fair.py 与 LBP-BoW 的双分支融合

本文件夹在**不改动 classifyML 任何原始文件**的前提下，把
[`train_cle_fair.py`](../train_cle_fair.py) 的「公平训练协议」与
[BoW 项目](../../BoW/train_dual_fixed.py) 的「LBP-BoW 双流分支」融合成一个双分支网络，
并提供配套的「特征提取 + ML 分类器」检查脚本，**评价指标完全复用
[`evaluate.py`](../evaluate.py)**，保证与单流评估口径一致。

## 文件说明

| 文件 | 作用 | 复用 / 参照 |
|---|---|---|
| `bow_extractor.py` | 多尺度 LBP-BoW 提取 + pkl 缓存 + `BOWFeatureCache` | 改自 `BoW/bow_extractor.py` |
| `my_dataset_bow.py` | 返回 `(img, bow_feat, label)` 的数据集；`align_bow` 兼容 RandomCrop + RandomResizedCrop | 改自 `BoW/my_dataset_bow_fixed.py` |
| `dual_stream_model.py` | 通用双流融合模型：任意注册表骨干(去头) + BoW 纹理流 + Cross-Attention | 结构参照 `BoW/models/DualStreamNet_fixed.py`；骨干用 `cle_fair_common.build_model` |
| `train_dual_fair.py` | 双分支训练（val MACRO-F1 / run_meta / ImageNet 归一化 + BoW 分支） | 融合 `train_cle_fair.py` + `BoW/train_dual_fixed.py` |
| `extract_and_classify_dual.py` | 特征提取 + ML 分类器检查（指标对齐 evaluate.py） | 流程参照 `BoW/extract_and_classify_dual.py`；指标复用 `evaluate.py` |
| `dual_stream_resnet34_cpca.py` | 以 `resnet34_degconv_cpca` 为固定主干的专用双流模型（feat_dim=512，≈22.7M） | 复用 `dual_stream_model.DualStreamFusion`；骨干来自 `models/optimize/resnet34_CPCA.py` |
| `fasternet_cpca.py` | **FasterNet + CPCA** 新骨干 `fast_t0_cpca`（feat_dim=1280，骨干≈2.87M）：在 fast_t0 的深层 stage 后插入 CPCA 注意力，不改 `models/FasterNet.py` | 复用 `models.FasterNet.fast_t0` 子模块；CPCA 用用户提供的原始实现 |
| `fasternet_clecpca.py` | **FasterNet + CLECPCA** 新骨干 `fast_t0_clecpca`（feat_dim=1280，骨干≈3.12M）：用针对 CLE 改进的 CLECPCA（std 池化 + 3×3 细节分支 + softmax 自适应多尺度融合），与 `fast_t0_cpca` 并存便于消融 | 复用 `models.FasterNet.fast_t0` 子模块；CLECPCA 用用户提供的实现 |

### 复用的现有函数（直接 import，不复制）

- `cle_fair_common.build_model / MODEL_REGISTRY / build_val_transform`（`classifyML/cle_fair_common.py`）
- `utils.read_split_train_data / read_split_val_data`（`classifyML/utils.py`）
- `evaluate.compute_metrics / bootstrap_ci / print_report / save_outputs`（`classifyML/evaluate.py`）

## 运行约定

所有脚本都从 **classifyML 根目录** 运行（脚本顶部已自动把根目录加入 `sys.path`）：

```bash
cd /home/login/cle/sunfuxiang/classifyML
```

## 安装依赖

除 classifyML 既有依赖外，还需（已确认当前环境可用）：

```bash
pip install opencv-python scikit-image xgboost lightgbm
```

## 1. 训练双分支模型

```bash
python bow_fusion/train_dual_fair.py \
    --model fast_t0 --num_classes 3 --img_size 224 --seed 42 \
    --weights <backbone预训练权重.pth> \
    --data-path-train /path/to/cle/train \
    --data-path-val   /path/to/cle/val \
    --save-dir ./bow_fusion/weights \
    --epochs 300 --batch-size 32 --lr 1e-4 --early-stop 30
```

要点：
- **归一化**：直接用 ImageNet `[0.485,0.456,0.406]/[0.229,0.224,0.225]`（你加载预训练权重，无需自算 mean/std，对齐 `train_cle_fair.py` 的 `if args.weights` 分支）；可用 `--norm-mean/--norm-std` 覆盖。
- **选最优 / 早停**：按 **val MACRO-F1**（对齐 `train_cle_fair.py`）。
- **BoW 分支**：默认 `--align-bow` 关闭，使用预提取的「全图」BoW 缓存；加 `--align-bow` 改为实时提取「与 CNN 同一裁剪区」的 BoW（更严格，更慢）。BoW 参数 `--bow-K`(默认256)、`--bow-scales`(默认`16,32,64`)，维度 = `len(scales)*K = 768`。
- **骨干**：`--model` 可取 `cle_fair_common.MODEL_REGISTRY` 中任意模型（fast_t0 / resnet50 / vit_base_patch16_224 / swin_* …）。分类头会被自动移除，`feat_dim` 自动推断。
- **预训练权重**：`--weights` 既支持**单骨干**权重（自动重映射到 `backbone.*`、跳过 head/fc/classifier），也支持**既有双流**权重（跳过 classifier）。
- **冻结骨干**：加 `--freeze-backbone` 只训练融合头。
- 输出：`weights/dual_<model>/run_meta.json`、`model_best.pth`、`train_log.csv`（含 `val_macro_f1` 列）。

## 2. ML 检查（特征提取 + 多分类器对比）

```bash
python bow_fusion/extract_and_classify_dual.py \
    --weights bow_fusion/weights/dual_fast_t0/model_best.pth \
    --model fast_t0 --img_size 224 \
    --data-train /path/to/cle/train \
    --data-test  /path/to/cle/test \
    --class-names <类名1> <类名2> <类名3>     # 可选，不传则自动从训练目录检测
    # --bootstrap        # 可选：核心指标 95% Bootstrap CI
```

流程（与 `BoW/extract_and_classify_dual.py` 一致，指标改用 `evaluate.py`）：
1. 提取 / 缓存 train + test 的 LBP-BoW；
2. 构造 `dual_stream`，置 `classifier=Identity` 作特征提取器（特征维度 = `2*d_model`）；
3. 提取 train/test 特征存 `.npy`，`StandardScaler` 标准化；
4. 基准 **E2E**（整模型 `(img,bow)`→softmax）+ **SVM-RBF / SVM-Linear / LogisticRegression / KNN / RandomForest / XGBoost / LightGBM / Voting**；
5. 每个方法调用 `evaluate.compute_metrics`（+可选 `bootstrap_ci`）、`evaluate.save_outputs` 落盘 `predictions.csv / results.json / evaluation_report.md`；
6. 终端对比表 + `comparison_summary.csv`（macro-F1 / Accuracy / AUROC(OvR) / AUC-PR / Kappa / MCC / Balanced-Acc / 各类 Sens/Spec 等）。

> `--model`、`--img_size`、`--bow-K`、`--bow-scales` 必须与训练时一致；归一化自动从权重旁 `run_meta.json` 读取（与训练同口径）。

## 与单流评估的一致性

本文件夹的所有指标都来自 `classifyML/evaluate.py` 的 `compute_metrics / bootstrap_ci`，
因此双分支模型在此处的 macro-F1 / AUROC / AUC-PR / Kappa / MCC / 每类 Sens-Spec 与
对单流模型运行 `evaluate.py` 的口径**完全一致**，可直接横向比较。

## 3. 更换深度主干：resnet34_degconv_cpca（专用文件）

默认主干是 `fast_t0`。要换成 `models/optimize/resnet34_CPCA.py` 里的
**resnet34_degconv_cpca**（DEGConv 浅层 layer1+2 + CPCA 深层 layer3+4 的 ResNet34 改进臂，
feat_dim=512，整模型 ≈22.7M 参数），有两种等价方式：

**方式 A（最省事，推荐）** —— 直接给通用脚本传 `--model`（该骨干已在注册表中）：

```bash
python bow_fusion/train_dual_fair.py \
    --model resnet34_degconv_cpca --num_classes 3 --img_size 224 \
    --data-path-train /path/to/cle/train --data-path-val /path/to/cle/val \
    --save-dir ./bow_fusion/weights --epochs 300
# ML 检查同样传 --model resnet34_degconv_cpca（须与训练一致）
```

**方式 B** —— 用专用文件 `dual_stream_resnet34_cpca.py`（把 resnet34_degconv_cpca 锁定为
深度主干的命名入口，便于复制/自检）。自检：

```bash
python bow_fusion/dual_stream_resnet34_cpca.py   # 打印 feat_dim / 参数量 / 前向+反向
```

如需把专用文件作为 `create_model` 嵌入训练/评估，把入口脚本里的
`from dual_stream_model import dual_stream as create_model` 换成
`from dual_stream_resnet34_cpca import dual_stream_resnet34_cpca as create_model` 即可
（接口签名一致：`create_model(num_classes=..., bow_dim=...)`）。

> 骨干头自动移除：resnet34_degconv_cpca 的分类头是 `self.fc`（`include_top=True` 默认），
> 被通用 `_strip_classifier` 识别为 `fc` 并置 `Identity`，forward 返回 `[B, 512]` 池化特征。
> 同理，注册表里其它骨干（resnet50 / vit / swin / densenet / repvit …）也都可直接用 `--model` 切换。

### 自建骨干：fast_t0_cpca（FasterNet + CPCA）

`fasternet_cpca.py` 在**不改 `models/FasterNet.py`** 的前提下，复用 `fast_t0` 的
`patch_embed / stages / avgpool_pre_head / head` 子模块，并在指定 stage 输出后插入用户提供的
**CPCA 注意力**（通道 avg+max 注意力 + 多尺度条带深度卷积空间注意力），作为新骨干 `fast_t0_cpca`。
CPCA 空间尺寸/通道数不变 → feat_dim 仍为 1280；fast_t0 预训练权重可直接载入同名子模块，
CPCA 为新增分支随机初始化后随训练学习（载入时会提示 `missing=62` 等，属正常）。

```bash
# 训练（深度流用 FasterNet+CPCA，BoW 分支不变）
python bow_fusion/train_dual_fair.py --model fast_t0_cpca --num_classes 3 --img_size 224 \
    --weights pth/fasternet_t0-epoch.281-val_acc1.71.9180.pth \
    --data-path-train /path/to/cle/train --data-path-val /path/to/cle/val \
    --save-dir ./bow_fusion/weights --epochs 300
# ML 检查同样传 --model fast_t0_cpca（须与训练一致）

# 自检（打印 feat_dim / 参数量 / 前向+反向 / state_dict 含 cpca.*）
python bow_fusion/fasternet_cpca.py --cpca_positions 2,3
```

> CPCA 插入位置由 `cpca_positions` 控制，默认 `(2, 3)`（深层 Stage2/Stage3，对应 resnet 把 CPCA
> 放在 layer3/4）；可改为 `(3,)` 只加一个、或 `(0,1,2,3)` 全阶段。`fast_t0_cpca` 已在 bow_fusion
> 本地扩展注册表中（不修改 `cle_fair_common.py`），故 `--model fast_t0_cpca` 可直接用。

### CLE 改进版：fast_t0_clecpca（FasterNet + CLECPCA）

`fasternet_clecpca.py` 与 `fasternet_cpca.py` 结构平行，但把注意力模块换成用户提供的
**CLECPCA**（针对 CLE 灰度图的改进 CPCA）：通道注意力增加 std(标准差)池化支路、3×3 深度卷积
细节/高频分支、softmax 自适应多尺度融合、独立卷积（不再共享 1×1）；超大核(21)默认关闭
（`use_large`）。feat_dim 仍为 1280。**`fast_t0_cpca`（原版 CPCA）与 `fast_t0_clecpca`（CLE 改进）
两个 `--model` 选项并存**，便于消融对比（CLECPCA 作者也建议逐项开关验证）。

```bash
# 训练（深度流用 FasterNet+CLECPCA，BoW 分支不变）
python bow_fusion/train_dual_fair.py --model fast_t0_clecpca --num_classes 3 --img_size 224 \
    --weights pth/fasternet_t0-epoch.281-val_acc1.71.9180.pth \
    --data-path-train /path/to/cle/train --data-path-val /path/to/cle/val \
    --save-dir ./bow_fusion/weights --epochs 300
# ML 检查同样传 --model fast_t0_clecpca（须与训练一致）

# 自检（可加 --use_large 启用 21x21 超大核分支）
python bow_fusion/fasternet_clecpca.py --cpca_positions 2,3
```

## 快速自检

```bash
# 通用模型构造与前向（打印 feat_dim / 输出形状）
python bow_fusion/dual_stream_model.py --model fast_t0 --num_classes 3 --bow_dim 768
# resnet34_degconv_cpca 专用自检（打印 feat_dim / 参数量 / 前向+反向）
python bow_fusion/dual_stream_resnet34_cpca.py

# 参数帮助
python bow_fusion/train_dual_fair.py --help
python bow_fusion/extract_and_classify_dual.py --help
```

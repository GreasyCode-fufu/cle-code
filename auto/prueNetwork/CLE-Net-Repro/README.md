# CLE-Net 可复现参考项目

本项目冻结已在本机完整复现的 CLE-Net 协议。CLE-Net = ConvNeXt-T + EGB + SCG，注册名 **convnext_egb_scg**；HSC属于消融变体，不是最终模型。

## 已验证结果

2026-11-09由Codex独立从ImageNet权重启动训练，seed42/43/44全部完成，固定CLE_ac测试集576张：

| 模型 | Seed42 | Seed43 | Seed44 | Macro-F1均值±样本标准差 |
|---|---:|---:|---:|---:|
| ConvNeXt-T baseline | 0.963587 | 0.959687 | 0.952851 | 0.9587±0.0054 |
| CLE-Net | 0.961691 | 0.958511 | 0.968832 | **0.9630±0.0053** |

CLE-Net三个检查点的293个状态张量、训练CSV、九项测试指标与2026-10-27修复版记录逐项完全一致。baseline与10月30日baseline_v2逐项一致。均值差为**+0.430323个百分点**，三个种子中一次为正；不能继续将旧baseline 0.9489±0.0122对应的+1.41pp写为本协议的收益。未作显著性检验。

本数值是下述固定条件的已验证结果，不是对任意环境或数据划分的性能保证。完整证据在 `reference_results/` 和 `reference_summary/`；checkpoint原始位置列于 `REFERENCE_INDEX.json`。

## 复现条件

| 项目 | 固定条件 |
|---|---|
| 历史源码 | 原Network/CLE-Net Git b457d93e13fca274876cfd7aa040d7db3a5aef15；核心.py逐字节保留，锁定于source_files.sha256.json |
| Python / PyTorch | sunfuxiang环境，Python3.12.0，torch2.12.0+cu130，torchvision0.27.0+cu130 |
| 依赖与硬件 | 详见environment/；本机RTX4090，CUDA运行库13.0；OMP和MKL线程各2 |
| 数据 | CLE_ac_v1：train2077、val427、test576；原数据逐文件哈希在environment/dataset_manifest.json |
| 类别 | Benign_lesions / Malignant_lesions / body_of_stomach_normal；测试支持数208/176/192 |
| 输入 / 归一化 | RGB读取，224×224；ImageNet mean=.485/.456/.406，std=.229/.224/.225 |
| 预训练 | convnext_tiny_1k_224_ema.pth，SHA256=14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595 |
| 新分类头 | 三分类，沿用冻结代码的nn.init.trunc_normal_(std=0.2)及torch2.12实现；不能仅换PyTorch版本而认为同seed初始化相同 |
| 优化器 | AdamW，普通参数lr=1e-4/wd=.05；15个EGB门控lr=.01/wd=0 |
| 训练 | batch32、最大300epochs、CosineAnnealingLR(T_max=300)、无warmup、未加权CE、label smoothing=0、AMP |
| checkpoint | 只按训练内验证Macro-F1选择；连续30轮无提升早停 |
| loader | 普通shuffle、3workers；无独立generator、无persistent_workers，保留历史约定 |
| 种子 | 42/43/44；Python/NumPy/Torch/CUDA设seed，cudnn.deterministic=True、benchmark=False |
| 统计 | 各seed独立测试后取均值和样本标准差(ddof=1)；不集成，不以测试集挑checkpoint |

训练增强依次为Resize((256,256))、水平/垂直翻转各p=.5、±15°旋转、RandomCrop(224)、归一化。训练内验证使用强制方形Resize((256,256))+CenterCrop；独立evaluate.py使用短边Resize(256)+CenterCrop。这里保留了历史两条路径的差异；若修订预处理，需要作为新协议重跑并另行命名。

## 使用

```bash
cd /home/login/data/cle/auto/prueNetwork/CLE-Net-Repro
# 只检查，不训练
bash run_reference.sh --check-only
# 明确启动：按seed依次运行baseline和CLE-Net，共六次训练
bash run_reference.sh --run
# 可选的独立消融批次示例
bash run_reference.sh --run --models convnext_egb convnext_scg --seeds 42 43 44
```

启动器会先验证环境、核心源码哈希、全数据哈希、预训练哈希和seed42分类头初值哈希。检测到错误环境（例如cle_topnets）将拒绝运行。每批创建新的 `runs/<时间戳>/`，保存命令、PID、退出码、训练/验证/测试记录和九项汇总指标；运行前如GPU有其他计算进程则等待，不终止别人的任务。

数据和预训练权重是外部共享资源，绝对路径保存在protocol_lock.json。本项目不复制大型数据或历史checkpoint；REFERENCE_INDEX.json指明已验证权重位置。不要运行旧项目中会删除输出目录的R6调度脚本来覆盖本项目。

## 目录和版本

- 顶层训练/评估文件及models/：冻结的历史源码。
- tools/preflight.py：环境、源码、数据、权重及初始化检查。
- tools/run_reference.py、run_reference.sh：带记录的统一入口。
- environment/：版本、pip freeze、数据清单、硬件记录。
- reference_results/：六次成功复现的日志、元数据、测试结果与预测副本。
- reference_summary/：均值/标准差、配对差和检查点复现审计。
- audit/：旧项目原始结果清点、修正后的统计与逐类使用建议。
- docs/EXPERIMENT_REUSE_AUDIT.md：论文哪些实验可复用、哪些需补跑。

项目已建立自己的Git历史。原始Network、旧prueNetwork/CLE-Net、codex和Zcode项目均未改动。未来修改核心代码会使校验失败；应建立新的协议和结果批次，不能覆盖此参考结果。

患者独立性尚未核实；原项目记录固定测试集参与过开发反馈。因此这些结果是固定图像集合的模型实验，不能据此直接主张患者独立或外部临床验证。

## 论文实验CSV汇总

`paper_results_csv/` 提供51个UTF-8-BOM CSV文件，覆盖CLE、Covid、BUSI、CUB、TCGA。先查看 `00_文件索引.csv`、`01_协议与环境.csv`、`29_统计与使用说明.csv`。

包含已验证主结果、传统骨干对比、2025/2026四种新方法及其同环境控制、TransNeXt2024参考、消融、跨数据集比较与差值、容量扩展、逐类指标、稀疏混淆矩阵、未完成状态、原始表副本。重算表统一采用样本标准差；不同协议和批次保留单独标记。原始副本不自动代表可直接用于统一主表。

生成器为 `tools/collect_paper_csv.py`，来源为原始results.json及审计清单。已校验索引哈希、行数、八组汇总/逐种子表的九项指标和标准差。该汇总不新增任何训练，也不把未完成实验记为零分。

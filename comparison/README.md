# 对比实验代码库（R6, 2026-11-09）

本目录集中存放论文对比实验所涉及方法的**官方代码库**（`third_party/`, 只读参考）与获取脚本。

- **获取方式**：本会话可用的 MCP 服务（文档 / PDF / 演示文稿 / 表格 / 图片搜索）均不支持代码库下载，
  因此用 `git` 经本机代理浅克隆（`fetch_repos.sh`）。每个仓库位于独立目录，精确 commit 与许可证见 `third_party/MANIFEST.tsv`。
- **使用原则**：官方代码只用于**阅读核对、保证忠实复现**，不在其中直接运行 —— 各官方实现的数据管线、
  优化器与评测协议互不相同；为保证公平，全部对比方法都在本仓库的统一协议内实现
  （AdamW lr=1e-4 wd=5e-2、cosine、bs=32、相同增广、val Macro-F1 选点、早停 30、seeds 42–46、evaluate.py 9 指标）。
- `third_party/` 不入 git（体积与第三方许可证）；需要时运行 `./fetch_repos.sh` 重新获取。

## 代码库与本仓库中的对应实现

| 方法 / 主干 | 论文 | 官方代码库 | 本仓库实现 | 复现要点（据官方代码核对） |
|---|---|---|---|---|
| LP-FT | Kumar et al., ICLR 2022 | AnanyaKumar/transfer_learning | `models/optimize/r6_ftbaselines.py::mbv3_lpft` + `experiments/r6_lpft_prepare.py` | `log_reg_sk.py`：特征全局标量归一化、`LogisticRegression(warm_start)`、C∈logspace(−7,2,100) 按 val 准确率选；`set_last_layer` 初始化头后全网络微调 |
| L2-SP | Li et al., ICML 2018 | holyseven/TransferLearningClassification | `r6_ftbaselines.py::mbv3_l2sp_a01 / a001` | `network_base.py`：仅对 `weights`（conv/fc 核）施加 α·½‖w−w⁰‖²（已有层）与 β·½‖w‖²（新层）；默认 α=β=0.01，另试论文 α=0.1 |
| Surgical fine-tuning | Lee et al., ICLR 2023 | anniesch/surgical-finetuning | `r6_ftbaselines.py::mbv3_sft_{b1,b2,b3,b4,head}` | `main.py`：只把所选块参数交给优化器；整体 `model.train()`（冻结块 BN 统计量仍更新）；块按分辨率对应 conv1+layer1/layer2/layer3/layer4/fc |
| cRT / τ-norm | Kang et al., ICLR 2020 | facebookresearch/classifier-balancing | `experiments/r6_crt.py` | `cls_crt.yaml`：冻结主干、重新初始化分类器、ClassAwareSampler、SGD(0.9, wd 5e-4)、lr 0.2@bs512（按批量缩放）、cosine 10 epoch；`tau_norm.py`：逐类 ‖w‖^τ 归一化、无偏置、τ∈[0,2] |
| ConvNeXt-T | Liu et al., CVPR 2022 | facebookresearch/ConvNeXt | `models/Convnext.py`；EPIC 版 `models/optimize/epic_generic.py` | 实验 2 跨主干 |
| StarNet-S2 | Ma et al., CVPR 2024 | ma-xu/Rewrite-the-Stars | timm `starnet_s2`（`models/timm_light.py`）；EPIC 版 `epic_generic.py` | 实验 2 跨主干 / 对比表 |
| FasterNet-T0 | Chen et al., CVPR 2023 | JierunChen/FasterNet | `models/FasterNet.py`；EPIC 版 `epic_generic.py` | 实验 2 跨主干 / 对比表 |
| GhostNet | Han et al., CVPR 2020 | huawei-noah/Efficient-AI-Backbones | `models/GhostNet.py` | 对比表 |
| MobileNetV4-Conv-S | Qin et al., ECCV 2024 | huggingface/pytorch-image-models（稀疏检出 `timm/models/mobilenetv3.py` 等） | timm `mobilenetv4_conv_small` | 对比表 |
| MobileNetV3-L / ShuffleNetV2 | Howard et al., ICCV 2019 / Ma et al., ECCV 2018 | pytorch/vision（稀疏检出） | `models/MobileNetV3.py` / `models/ShuffleNet.py` | 基座 / 对比表 |

注：L2-SP 检索结果给出的 `thuml/TransferLearningClassification` 不存在，改用论文第一作者账号 holyseven 下的仓库。

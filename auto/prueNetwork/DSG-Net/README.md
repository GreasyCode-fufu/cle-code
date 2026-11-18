# DSG-Net（ConvNeXt-T + 方向结构校准）：独立副本

> 项目目录 2026-10-30 由 `convnext_tiny_cle_oca_strip` 更名为 **DSG-Net**（注册名/代码不变）；历史文档中旧路径均指向本目录。

实际入口：`convnext_tiny_cle_oca_strip`。本目录的网络源码从原项目
`/home/login/data/cle/auto/codex/Classification`（git `d51211b`）逐字节复制；
源文件同时包含 DSG 家族的其它研究变体（dose/ptg/gds 等单点与组合，含负结果），
正式完整网络请使用本入口。

## 使用

```bash
conda activate sunfuxiang
cd /home/login/data/cle/auto/Network/DSG-Net
python smoke_test.py   # 前向/参数量/严格加载/与baseline逐位恒等 全部断言
```

```python
import torch
from model import build_model

model = build_model(num_classes=3)
x = torch.randn(1, 3, 224, 224)
logits = model(x)  # [1, 3]；未经 softmax 的分类 logits
```

输入使用 ImageNet 归一化 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，224×224。
训练侧增强与评测预处理与原协议一致（训练 Resize(256,256)+翻转/旋转15°+RandomCrop224；
评测 Resize(256)→CenterCrop(224)）。

## 文件作用

- `model.py`：便捷入口，默认 3 类（CUB 等 200 类传 num_classes=200）。
- `models/optimize/convnext_cle_dsg.py`：DSG 家族源码（OCA/DOSE 条带/PTG/GDS 及组合工厂）。
- `models/Convnext.py`：原 ConvNeXt 主干、LayerNorm、DropPath（基类）。
- `docs/ARCHITECTURE.md`：创新说明（机制、数学、灵感来源、消融与验证结论）。
- `docs/CAM_OVERLAY_EXPERIMENTS.md`：CAM 热图叠加实验记录（自种子类加权 CAM +0.82pp 3/3 及待办）。
- `docs/SELECTED_MODEL.json`：三种子最佳权重路径/SHA256 与 cle_ac 及跨域全指标。
- `COPY_MANIFEST.json`：复制来源、git 版本、各源文件 SHA256；`VERIFICATION.json`：
  独立导入/前向/恒等性检查结果（smoke_test.py 输出）。

本网络仅依赖 PyTorch 与标准库。目录已单独初始化 Git。

## 训练配方（单阶段，与 baseline 完全同协议）

ImageNet ema 预训练（`convnext_tiny_1k_224_ema.pth`, sha256 14f3164e...）严格加载
（允许缺失 head.* 与新分支段，0 unexpected），AdamW lr 1e-4 / wd 5e-2、bs 32、
≤300 epoch、Cosine(T_max=300)、AMP、早停 30、val macro-F1 选点、seeds 42/43/44。
两个创新模块零初始化：载入预训练后初始函数与 baseline **逐位一致**，训练中渐进解锁。
训练启动程序为原仓库 `train_cle_fair.py`（`--model convnext_tiny_cle_oca_strip`），
评测为 `evaluate.py` 全指标口径。

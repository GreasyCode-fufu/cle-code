# EF-ConvNeXt — 高效频域 ConvNeXt 用于共聚焦激光显微内镜(CLE)图像分类

交付版本: 轻量版 EF-ConvNeXt (`convnext_ef`) · 2026-11-02 · 源仓库
`/home/login/data/cle/auto/ZCode/Classification` (commit `9afe1eb`)

## 1. 一句话

在小样本 CLE 三分类(良性/恶性/正常)上, 以 **Fourier 对偶 + Eckart-Young 低秩**
两条数学主线重构 ConvNeXt-T: **参数 -34.3% (28.06M→18.42M)**, 主域 5seeds
Macro-F1 **95.48±0.39 (+0.72)** / Kappa **+1.10** / MCC +1.07, 五域跨域
**3 正 + 1 近无损 + 1 边界**。

## 2. 结构与三创新点

```
输入 224×224
 └─ ConvNeXt-T 主干
     ├─ stage1-2: 原样(预训练 100% 加载)
     ├─ stage3:   MLP → SVD 低秩折叠(p·q 双线性, r=C/2)     ← 创新点2
     └─ stage4:   dwconv → GFC 分组频域全局卷积(rfft2 滤波)   ← 创新点1
                  + MLP → SVD 低秩折叠
 └─ 分类头: Linear(768→3)
```

- **创新点1 GFC 分组频域全局卷积**: 7×7 dwconv → rfft2 可学习滤波(空域循环
  卷积的频域对偶), 感受野 7×7→全图; 8 组通道共享(参数 86K→11K, 频域先验正则);
  初始化 = 预训练核循环移位后 rfft2(与原算子数学等价, 预训练语义无损)。
- **创新点2 SVD 低秩折叠 MLP**: 占全网 90% 参数的 4C 扩张 MLP 折叠为 r=C/2
  双线性层; 初始化 = 预训练权重 SVD 的 Eckart-Young 最优低秩近似
  (W ≈ (U_r Σ_r)·V_rᵀ)。小样本域呈"低秩即正则"效应(-34% 参数反而 +0.91)。
- **创新点3 容量自适应协同架构(机制)**: 谱分析证明两域权重谱集中度相同
  (top-50% 秩能量 84.1% vs 84.1%), 低秩增益差异源自任务容量需求
  (小样本少类=正则增益, 大样本多类=容量瓶颈); GFC 分组正则在异质域对冲
  SVD 负贡献; 位置矩阵证明频域算子仅宜深层(stage4 95.5 vs stage3 93.8)。

## 3. 成绩单

### 主域 cle_ac_v1 (test 576 张, 5 seeds 42-46)

| 指标 | ConvNeXt-T | **EF-ConvNeXt** | Δ |
|---|---|---|---|
| Macro-F1 | 94.76±0.84 | **95.48±0.39** | **+0.72** |
| Cohen's Kappa | 92.16±1.29 | **93.26±0.56** | **+1.10** |
| MCC | 92.22±1.25 | 93.29±0.54 | +1.07 |
| Balanced Acc | 94.70±0.92 | 95.45±0.43 | +0.75 |
| Params | 28.06M | **18.42M** | **-34.3%** |
| MACs | 4.47G | **3.41G** | **-23.7%** |
| 最优 run (seed43) | — | Macro-F1 **95.98** / Kappa 94.03 | — |

### 跨域 (vs 各域 convnext_tiny 同协议 3 seeds)

| 域 | Δ Macro-F1 | 说明 |
|---|---|---|
| **CLE 主域** | **+0.72** | 5seeds |
| **Covid CT** | **+0.53** (98.55) | 小样本域 |
| **BUSI 乳腺超声** | **+0.14** (86.92) | GFC 正则对冲 SVD 负贡献 |
| **heelBone X光** | **-0.29** (87.66) | 近无损(此前所有变体 -1~-2) |
| CUB-200 鸟类 | -4.70 | 大数据域 SVD 容量瓶颈(适用边界, 诚实报告) |

## 4. 目录说明

```
EF-ConvNeXt/
├── README.md                 本文件
├── INNOVATIONS_AND_RESULTS.md  创新点详解+五域完整指标(论文取用)
├── demo.py                   独立推理演示(仅需 PyTorch/torchvision/PIL)
├── requirements.txt
├── models/
│   ├── Convnext.py           baseline 主干
│   └── optimize/convnext_ep.py   EF-ConvNeXt 实现(FGC/GFC/FoldedLinear)
├── weights/
│   ├── ef_convnext_best.pth  最优权重(seed43, Macro-F1 95.98)
│   └── convnext_ef_seed4{2-6}.pth  全部 5 种子(多种子统计)
├── eval/
│   ├── evaluate.py           完整评估脚本(12 指标+混淆矩阵+bootstrap)
│   └── my_dataset.py
├── jobs/                     复现实验清单(主域+跨域, 配源仓库 runner.py)
├── paper_data/               全部实验原始 CSV
└── reports/
    ├── FINAL_EF_REPORT.md    终版报告(结构/成绩/机制/消融)
    ├── FINAL_NETWORK_REPORT.md  精度版 FD-ConvNeXt 对照报告
    ├── EPNET_DESIGN.md       设计文档(数据驱动分析+27变体探索矩阵)
    └── cle_image_analysis.*  CLE 图像物理特性分析(创新点动机数据)
```

## 5. 快速使用

```bash
# 独立推理演示
python demo.py --image <一张CLE图路径>
python demo.py                     # 内置示例前向

# 完整评估(test 集)
cd eval && python evaluate.py --model convnext_ef \
    --weights ../weights/ef_convnext_best.pth \
    --data-path <cle_ac_v1/test 路径> --bootstrap

# 训练复现(源仓库 /home/login/data/cle/auto/ZCode/Classification)
python experiments/runner.py experiments/jobs_batchBQ.json 3
# 跨域: RUN_DATA=datasets/covid_v1 RUN_NCLS=200 等, 详见 jobs/
```

归一化: ImageNet mean/std(预训练一致); 类序: sorted 目录
[Benign_lesions, Malignant_lesions, body_of_stomach_normal]。

## 6. 精度版参照

同源交付的精度版 FD-ConvNeXt(`convnext_fgc_dsd`, 27.67M, F1 95.58/峰值 97.04)
见源仓库 experiments/FINAL_NETWORK_REPORT.md; 两版共享 FGC 核心, 论文可双表呈现。

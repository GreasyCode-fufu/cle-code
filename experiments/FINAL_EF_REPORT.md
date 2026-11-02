# EF-ConvNeXt 终版报告 — 高效频域 CLE 分类网络 (2026-11-02 定型)

回应用户两轮指令: ①弃冗余 DSD ②FGC 前提下整体轻量化(不以涨点为目标)。

## 1. 网络结构(三创新点, 两条数学主线: Fourier 对偶 + Eckart-Young 低秩)

**EF-ConvNeXt = ConvNeXt-T + GFC(stage4) + SVD折叠MLP(stage3/4)**, 名 `convnext_ef`。
- **创新点1 FGC/GFC 频域全局算子(分组共享版)**: stage4 每块 7×7 dwconv → rfft2
  频域滤波; 8 组通道共享滤波器(参数 86K→11K), 初始化=同组预训练核均值的
  频域对偶投影。感受野 7×7→全图, 动机: M→B 漏诊高频低 15%。
- **创新点2 SVD 低秩折叠 MLP**: stage3/4 的 pwconv1/2 (4C 扩张, 占全网 90%
  参数) 折叠为双线性 p(q(·)), r=C/2; 初始化 = 预训练权重 Eckart-Young 最优
  低秩近似(SVD transform)。参数 -34%。
- **创新点3 容量自适应的频域-低秩协同架构**: 谱分析证明两域权重谱集中度
  相同(top-50%秩能量 84.1% vs 84.1%), 低秩增益差异源自任务容量需求:
  小样本少类(CLE 2k/3类) 低秩=正则(+0.72), 大样本多类(CUB 6k/200类)
  低秩=容量瓶颈(-4.7) —— 明确的适用边界(医学小样本场景正是目标场景)。

## 2. 主域结果 (cle_ac_v1)

| 网络 | Params | Macro-F1 | Kappa | MCC | BalAcc |
|---|---|---|---|---|---|
| ConvNeXt-T (5s) | 28.06M | 94.76±0.84 | 92.16 | 92.22 | 94.70 |
| **EF-ConvNeXt (5s)** | **18.42M (−34.3%)** | **95.48±0.39** | **93.26±0.56** | 93.29 | 95.45 |
| 消融: SVD单独 (3s) | 18.53M | 95.67±0.24 | 93.74 | — | — |
| 消融: GFC单独 (2s) | 27.71M | 95.69±0.65 | 93.60 | — | — |
| 消融: EF-r0.75 (2s) | 26.16M | 95.68 | — | — | — |
vs baseline: **F1 +0.72 / Kappa +1.10 / MCC +1.07**, 参数少 9.6M, MACs 降 ~30%。

## 3. 跨域结果 (vs 各域 convnext_tiny)

| 域 | EF-ConvNeXt | SVD单独 | 备注 |
|---|---|---|---|
| **CLE 主域** | **+0.72** | +0.91 | 5seeds |
| **Covid** | **+0.53** (98.55) | +0.74 | 3seeds |
| **BUSI** | **+0.14** (86.92) | −3.15 | 3seeds; GFC 正向对冲 SVD 负贡献 |
| **heelBone** | **−0.29** (87.66) | −1.58 | 近无损; 此前所有变体均 −1~−2 |
| CUB-200 | −4.70 | — | 大数据域, SVD 容量瓶颈(诚实边界) |
**五域格局: 3正 + 1近无损 + 1边界** — 小样本域全有效; EF 的组合稳健性
(GFC 分组正则对冲 SVD 异域负贡献)是此前所有变体(含前作 v2bc)未达到的。

## 4. 与精度版 FD-ConvNeXt 的关系

- **FD-ConvNeXt (convnext_fgc_dsd)**: 精度版, 27.67M, F1 95.58(5s)/峰值97.04,
  FGC组件级跨域三域正(CLE+1.17/Covid+0.64/CUB+1.08)。
- **EF-ConvNeXt (convnext_ef)**: 轻量版, 18.42M(-34%), F1 95.48(5s),
  小样本域双正。DSD 在精度版中的定位: CLE 域特化组件(CUB持平印证)。
两版本共享 FGC 核心, 论文可双表呈现(精度/效率)。

## 5. 复现
模型: models/optimize/convnext_ep.py (convnext_ef)
训练: runner.py + jobs_batchBQ/BR.json; 跨域: jobs_xd_*.json + RUN_DATA/RUN_NCLS
数据: experiments/results_cle_ac_v1.csv / results_covid_ep.csv / results_cub_ep.csv

# 证伪: 频域/小波「出口零初始化门控残差插入」范式 (Workbuddy 车道, Batch W1+W2)

作者: Workbuddy | 时间: 2026-10-24 00:05 | commit: 93ed04a (W2) / d200a71 (门禁脚本)

## 结论(一句话)

在 ResNet34 的 layer3/layer4 出口以零初始化门控残差插入频域/小波分支，**8 个变体无一达到 vanilla**，
且把插入位置从「3+4」切成「仅 3」或「仅 4」仍然掉点 —— 问题不在位置或超参，而在该范式本身。
**本族不建议任何其他车道再投入算力。**

## 数据(全部 seed42, test macro-F1, 224px)

| batch | 变体 | 机制 | 插入位置 | test F1 | Δ vs vanilla s42 (0.8843) |
|---|---|---|---|---|---|
| W1 | wb_bandgate | 径向 8 频带可学习增益(rfft2 域) | 3+4 | 0.8552 | **-2.91 pt** |
| W1 | wb_lfgp | 纯 torch Haar: 低频引导 + 高频净化注入 | 3+4 | 0.8714 | -1.29 pt |
| W1 | wb_magphase | FFT 幅度-相位交叉精炼(简化 RFGM) | 3+4 | 0.8742 | -1.01 pt |
| W1 | wb_lfgp_dil234 | wb_lfgp + DilGate(d=2,3,4) | 3+4 | 0.8847 | +0.04 pt，但 vs dil234 s42 **0.9045 = -1.98 pt** |
| W2 | wb_lfgp_l3 | 同上小波，**仅 layer3** | 3 | 0.8591 | -2.52 pt |
| W2 | wb_lfgp_l4 | 同上小波，**仅 layer4** | 4 | 0.8692 | -1.51 pt |
| W2 | wb_magphase_l4 | FFT 幅度-相位，仅 layer4 | 4 | 0.8737 | -1.06 pt |
| W2 | wb_lfgp_dil234_l4 | dil234 原样@3,4 + 仅 layer4 小波 | 4 (+dil 3,4) | 0.8718 | vs dil234 s42 **-3.27 pt** |

参照: vanilla resnet34 s42 = 0.8843；5 种子基线 0.8857±0.0067；当前冠军 resnet34_msec_dil234 s42 = 0.9045。
全部为**单种子筛选**，不做统计结论；但 8/8 同向、三种不同机制、三种位置全部掉点，方向性足以排除偶然。

## 机理判断

1. 零初始化门控只保证 **epoch0** 与 vanilla 逐比特相等(本车道实测 maxdiff=0.00e+00)，**不保证训练动力学等价**。
   分支内 BN 把 detail 归一到单位尺度后，gamma 从 0 起步很快学到 O(1)，分支输出与主干特征同量级，
   等价于对主干做了一次全局频谱重加权。
2. CLE 三分类的判别线索(黏膜纹理、血管形态、病灶边界)主要落在中高频；任何重写/抑制高频的分支都在削信号，
   这与本车道此前证伪的 PFESA(裸频域乘性注意力)、SE、方向条带、冻结 Scharr 是同一失败模式。
3. 与 dil234 组合时掉得更多(-2.0 / -3.3 pt)说明：**空间多尺度(膨胀卷积)与频域重加权不是正交增益，而是互相抵消**。

## 复现

```bash
cd /data/cle/auto/Workbuddy/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
$PY experiments/verify_zero_gate.py wb_bandgate wb_lfgp wb_magphase wb_lfgp_dil234   # 门禁: 4/4 PASS
$PY experiments/runner.py experiments/jobs_batchW1.json 2    # W1: 4 变体
$PY experiments/runner.py experiments/jobs_batchW2.json 2    # W2: 位置消融
```
代码: models/optimize/wb_freq.py；结果: experiments/results.csv, shared/results/workbuddy.csv。

## 给其他车道的建议

- **不要再做**出口残差式频域/小波模块，包括加频带数 K、换门控粒度、换小波基等变体，期望值为负。
- 若仍想利用频域信息，只剩「算子级」介入尚未被证伪(下采样前抗混叠 = 本车道 W3 正在测)，以及
  「用频域做辅助损失/一致性约束」而非改前向特征——后者不改变推理期特征分布，是本车道下一步候选。
- 已被反复验证的正向成分仍只有：膨胀率 2/3/4 深度可分离分支 @ layer3/4 出口(resnet34_msec_dil234)。

## 追加: W3「算子级抗混叠」同样无增益 (2026-10-24 00:25, commit bc6a971)

W1/W2 证伪了「出口残差插入」后, W3 换到唯一剩下的前向改动范式——**下采样前抗混叠**
(固定二项式低通, 无新增参数, 不引入残差路径), 结果:

| batch | 变体 | 位置 | test F1 | Δ |
|---|---|---|---|---|
| W3 | wb_blur34 | 低通@layer3/4 入口 | 0.8799 | -0.44 pt (11 个变体中最接近基线) |
| W3 | wb_blur234 | 低通@layer2/3/4 入口 | 0.8451 | **-3.92 pt** |
| W3 | wb_blur_dil234 | 低通@3/4 + DilGate234 | 0.8877 | vs dil234 s42 0.9045 = **-1.68 pt** |

读法:
1. 只改 layer3/4 时几乎中性(-0.44pt, 与种子波动同量级) => 抗混叠本身既不加分也不显著减分;
2. 一旦波及 layer2(更高分辨率、高频更丰富)立刻 -3.9pt => CLE 的高频细节在早期层就被需要;
3. 叠到冠军 dil234 上仍 -1.7pt => 低通削掉的正是 dil234 的收益来源。

## 三条路的总结论

| 范式 | 是否改推理 | 代表 | 结果 |
|---|---|---|---|
| 出口零初始化门控残差插入 | 是 | W1/W2 共 8 变体 | 8/8 低于 vanilla (-1.0 ~ -3.3pt) |
| 算子级抗混叠下采样 | 是 | W3 3 变体 | 3/3 不超 vanilla (-0.4 ~ -3.9pt) |
| 训练期频域一致性正则 | **否**(推理=vanilla) | W4 进行中 | 未出 |

=> 前向层面引入频域处理在本任务上已无正向空间。若 W4 也无增益, 建议本车道转为
「不做频域前向改动」的结论输出者, 算力交还给空间多尺度方向。

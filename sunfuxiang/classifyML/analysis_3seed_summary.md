# 三分类模型 3-seed 测试结果汇总（均值±标准差，含 Ours 与 CUFIT）

**实验设置**
- 原基线模型 6 个 + **Ours（ResNet34+DegConv+CPCA，本工作）** + **CUFIT（DINOv2 ViT-S/14 + 双 Rein）**
- 数据集：`full`=3class 原始　|　`hq`=3class_highquality 高质量；**Ours 仅报告 HQ 训练结果，CUFIT 仅报告 Full 训练的 3-seed 结果**；各组在同一独立 test 集评估，未报告项以 “—” 表示
- 随机种子 42/43/44，每个已报告的模型/训练数据组合各 3 次；统计为 **均值±样本标准差(ddof=1)**，百分制(%)，N=671
- <span style="color:red">**红色数字 = HQ 相对 Full 的提升（单位：百分点）**</span>
- CUFIT：Full 训练集 4977 张；三方共识 HQ 训练集 2880 张；共享 val=538、test=671。seed 42 复测后复用，43/44 于 2026-10-22 补跑。详见[完整对比报告](cufit_comparison_results/COMPARISON_REPORT.md)。

## 表1　主指标对比（Full vs HQ）

| 模型 | Full F1 | Full Acc | Full AUROC | HQ F1 | HQ Acc | HQ AUROC | ΔF1(HQ−Full) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ResNet34 | 87.64±0.23 | 87.48±0.26 | 96.52±0.09 | 90.11±0.98 | 89.97±1.01 | 98.02±0.14 | <span style="color:red">+2.48</span> |
| FasterNet-T0 | 87.44±1.05 | 87.38±0.99 | 96.48±0.39 | 89.99±0.28 | 89.87±0.26 | 97.65±0.25 | <span style="color:red">+2.56</span> |
| ViT-B/16 | 86.76±0.60 | 86.49±0.48 | 96.37±0.86 | 88.61±0.86 | 88.47±0.90 | 97.51±0.25 | <span style="color:red">+1.86</span> |
| Swin-T | 87.50±0.36 | 87.33±0.39 | 97.07±0.30 | 90.16±0.80 | 90.01±0.79 | 97.76±0.42 | <span style="color:red">+2.66</span> |
| PVT-Tiny | 87.33±1.43 | 87.18±1.42 | 96.83±0.89 | 90.13±0.49 | 89.97±0.48 | 97.64±0.19 | <span style="color:red">+2.81</span> |
| RepViT-M1 | 78.03±0.39 | 77.84±0.17 | 91.74±0.54 | 85.45±2.98 | 85.44±2.93 | 95.75±1.10 | <span style="color:red">+7.42</span> |
| **Ours** | — | — | — | 91.05±1.00 | 90.91±1.07 | 98.03±0.38 | — |
| **CUFIT** | 90.43±0.38 | 90.31±0.39 | 97.97±0.03 | — | — | — | — |

## 表2　HQ 相对 Full 各指标提升（<span style="color:red">**红字，单位百分点**</span>）

Δ = HQ均值 − Full均值。Ours 无 Full 版本、CUFIT 无 HQ 三种子版本，故为 “—”。基线平均仅统计原有 6 个具备配对结果的模型。

| 模型 | Macro-F1 | Accuracy | AUROC | Weighted-F1 | Balanced-Acc | Macro-Prec | Macro-Sens | Macro-Spec | AUC-PR | Kappa | MCC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ResNet34 | <span style="color:red">+2.48</span> | <span style="color:red">+2.48</span> | <span style="color:red">+1.51</span> | <span style="color:red">+2.53</span> | <span style="color:red">+2.52</span> | <span style="color:red">+2.62</span> | <span style="color:red">+2.52</span> | <span style="color:red">+1.30</span> | <span style="color:red">+3.23</span> | <span style="color:red">+3.75</span> | <span style="color:red">+3.85</span> |
| FasterNet-T0 | <span style="color:red">+2.56</span> | <span style="color:red">+2.48</span> | <span style="color:red">+1.17</span> | <span style="color:red">+2.57</span> | <span style="color:red">+2.68</span> | <span style="color:red">+2.26</span> | <span style="color:red">+2.68</span> | <span style="color:red">+1.34</span> | <span style="color:red">+2.06</span> | <span style="color:red">+3.79</span> | <span style="color:red">+3.73</span> |
| ViT-B/16 | <span style="color:red">+1.86</span> | <span style="color:red">+1.99</span> | <span style="color:red">+1.15</span> | <span style="color:red">+1.90</span> | <span style="color:red">+1.84</span> | <span style="color:red">+1.74</span> | <span style="color:red">+1.84</span> | <span style="color:red">+0.97</span> | <span style="color:red">+2.20</span> | <span style="color:red">+2.97</span> | <span style="color:red">+2.90</span> |
| Swin-T | <span style="color:red">+2.66</span> | <span style="color:red">+2.68</span> | <span style="color:red">+0.69</span> | <span style="color:red">+2.71</span> | <span style="color:red">+2.65</span> | <span style="color:red">+2.30</span> | <span style="color:red">+2.65</span> | <span style="color:red">+1.37</span> | <span style="color:red">+0.92</span> | <span style="color:red">+4.04</span> | <span style="color:red">+3.86</span> |
| PVT-Tiny | <span style="color:red">+2.81</span> | <span style="color:red">+2.78</span> | <span style="color:red">+0.82</span> | <span style="color:red">+2.84</span> | <span style="color:red">+2.89</span> | <span style="color:red">+2.54</span> | <span style="color:red">+2.89</span> | <span style="color:red">+1.46</span> | <span style="color:red">+1.05</span> | <span style="color:red">+4.22</span> | <span style="color:red">+4.14</span> |
| RepViT-M1 | <span style="color:red">+7.42</span> | <span style="color:red">+7.60</span> | <span style="color:red">+4.01</span> | <span style="color:red">+7.46</span> | <span style="color:red">+7.68</span> | <span style="color:red">+7.33</span> | <span style="color:red">+7.68</span> | <span style="color:red">+3.88</span> | <span style="color:red">+7.81</span> | <span style="color:red">+11.51</span> | <span style="color:red">+11.60</span> |
| **Ours** | — | — | — | — | — | — | — | — | — | — | — |
| *基线平均* | <span style="color:red">+3.30</span> | <span style="color:red">+3.34</span> | <span style="color:red">+1.56</span> | <span style="color:red">+3.34</span> | <span style="color:red">+3.38</span> | <span style="color:red">+3.13</span> | <span style="color:red">+3.38</span> | <span style="color:red">+1.72</span> | <span style="color:red">+2.88</span> | <span style="color:red">+5.05</span> | <span style="color:red">+5.01</span> |
| **CUFIT** | — | — | — | — | — | — | — | — | — | — | — |

## 表3　补充总体指标（FULL）

| 模型 | Weighted-F1 | Balanced-Acc | Macro-Prec | Macro-Sens | Macro-Spec | AUC-PR | Cohen κ | MCC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ResNet34 | 87.41±0.29 | 87.91±0.24 | 87.51±0.31 | 87.91±0.24 | 93.71±0.15 | 93.35±0.23 | 81.17±0.38 | 81.24±0.40 |
| FasterNet-T0 | 87.25±1.06 | 87.48±1.04 | 87.68±0.92 | 87.48±1.04 | 93.55±0.53 | 93.76±0.50 | 80.95±1.51 | 81.07±1.43 |
| ViT-B/16 | 86.57±0.57 | 86.93±0.70 | 87.14±1.01 | 86.93±0.70 | 93.24±0.29 | 93.46±1.48 | 79.69±0.76 | 79.97±0.98 |
| Swin-T | 87.29±0.36 | 87.73±0.76 | 87.83±0.14 | 87.73±0.76 | 93.63±0.33 | 94.97±0.78 | 80.94±0.67 | 81.22±0.67 |
| PVT-Tiny | 87.11±1.47 | 87.53±1.69 | 87.63±1.03 | 87.53±1.69 | 93.53±0.84 | 94.64±1.45 | 80.70±2.21 | 80.94±2.15 |
| RepViT-M1 | 77.86±0.40 | 77.95±0.37 | 78.39±0.91 | 77.95±0.37 | 88.76±0.09 | 84.60±1.19 | 66.56±0.25 | 66.67±0.27 |
| **CUFIT** | 90.27±0.39 | 90.67±0.42 | 90.38±0.29 | 90.67±0.42 | 95.14±0.20 | 96.28±0.05 | 85.43±0.59 | 85.52±0.54 |

## 表4　补充总体指标（HQ，含 Ours）

| 模型 | Weighted-F1 | Balanced-Acc | Macro-Prec | Macro-Sens | Macro-Spec | AUC-PR | Cohen κ | MCC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ResNet34 | 89.95±1.00 | 90.43±0.87 | 90.13±0.89 | 90.43±0.87 | 95.00±0.47 | 96.58±0.22 | 84.92±1.50 | 85.10±1.39 |
| FasterNet-T0 | 89.82±0.26 | 90.16±0.44 | 89.94±0.23 | 90.16±0.44 | 94.89±0.18 | 95.81±0.49 | 84.74±0.41 | 84.80±0.45 |
| ViT-B/16 | 88.47±1.00 | 88.77±0.57 | 88.88±0.82 | 88.77±0.57 | 94.21±0.40 | 95.66±0.51 | 82.66±1.29 | 82.86±1.09 |
| Swin-T | 90.00±0.79 | 90.38±0.77 | 90.13±0.71 | 90.38±0.77 | 95.00±0.37 | 95.89±0.81 | 84.98±1.17 | 85.08±1.10 |
| PVT-Tiny | 89.95±0.52 | 90.42±0.25 | 90.17±0.34 | 90.42±0.25 | 94.99±0.17 | 95.69±0.46 | 84.92±0.68 | 85.08±0.47 |
| RepViT-M1 | 85.32±3.01 | 85.64±2.79 | 85.72±3.14 | 85.64±2.79 | 92.64±1.45 | 92.41±1.96 | 78.07±4.39 | 78.26±4.38 |
| **Ours** | 90.91±1.04 | 91.29±0.91 | 91.03±0.91 | 91.29±0.91 | 95.46±0.51 | 96.45±0.84 | 86.33±1.60 | 86.45±1.49 |

## 表5　每类指标（FULL）— Sens/Spec/Prec/F1/AUROC

| 模型 | 类别 | Sens | Spec | Prec | F1 | AUROC |
|:--|:--|--:|--:|--:|--:|--:|
| ResNet34 | Benign lesions | 82.68±2.48 | 93.01±1.82 | 88.04±2.54 | 85.23±0.89 | 95.14±0.07 |
| ResNet34 | Malignant lesions | 83.33±1.26 | 91.32±0.75 | 81.41±1.19 | 82.35±0.58 | 94.82±0.19 |
| ResNet34 | body of stomach normal | 97.72±1.85 | 96.78±0.98 | 93.08±1.89 | 95.32±0.81 | 99.59±0.13 |
| FasterNet-T0 | Benign lesions | 87.11±1.79 | 89.64±2.30 | 83.92±2.74 | 85.45±1.09 | 94.79±0.58 |
| FasterNet-T0 | Malignant lesions | 77.94±4.05 | 94.58±0.99 | 86.80±1.80 | 82.07±1.99 | 95.19±0.66 |
| FasterNet-T0 | body of stomach normal | 97.40±0.75 | 96.42±0.81 | 92.32±1.55 | 94.78±0.57 | 99.46±0.08 |
| ViT-B/16 | Benign lesions | 80.99±3.63 | 93.73±1.97 | 88.98±2.66 | 84.72±1.06 | 95.03±1.04 |
| ViT-B/16 | Malignant lesions | 88.10±4.15 | 88.14±1.94 | 77.29±1.98 | 82.27±0.64 | 94.98±1.22 |
| ViT-B/16 | body of stomach normal | 91.71±4.95 | 97.85±1.76 | 95.14±3.60 | 93.28±1.70 | 99.08±0.34 |
| Swin-T | Benign lesions | 82.81±5.78 | 92.77±4.80 | 88.18±6.04 | 85.14±0.45 | 95.79±0.32 |
| Swin-T | Malignant lesions | 84.44±7.86 | 91.47±4.17 | 82.48±5.94 | 83.08±1.87 | 96.02±0.66 |
| Swin-T | body of stomach normal | 95.93±2.46 | 96.64±2.35 | 92.85±4.49 | 94.28±1.22 | 99.40±0.24 |
| PVT-Tiny | Benign lesions | 83.46±5.76 | 91.97±5.65 | 87.23±7.40 | 84.99±1.91 | 95.73±1.37 |
| PVT-Tiny | Malignant lesions | 83.02±9.11 | 91.68±3.31 | 82.42±4.25 | 82.36±2.76 | 95.48±1.22 |
| PVT-Tiny | body of stomach normal | 96.10±0.49 | 96.92±0.54 | 93.23±1.10 | 94.64±0.52 | 99.28±0.10 |
| RepViT-M1 | Benign lesions | 77.21±4.66 | 84.82±1.97 | 75.90±1.26 | 76.47±1.62 | 89.86±0.23 |
| RepViT-M1 | Malignant lesions | 70.63±4.77 | 86.04±1.54 | 69.79±1.21 | 70.14±2.16 | 87.79±0.72 |
| RepViT-M1 | body of stomach normal | 86.02±7.40 | 95.42±1.95 | 89.47±3.28 | 87.49±2.47 | 97.56±0.68 |
| **CUFIT** | Benign lesions | 86.33±3.34 | 94.86±1.95 | 91.32±2.72 | 88.69±0.63 | 97.48±0.16 |
| **CUFIT** | Malignant lesions | 87.62±4.54 | 93.06±1.72 | 85.31±2.47 | 86.36±1.29 | 96.81±0.04 |
| **CUFIT** | body of stomach normal | 98.05±0.84 | 97.50±0.33 | 94.52±0.69 | 96.25±0.61 | 99.62±0.03 |

## 表6　每类指标（HQ，含 Ours）— Sens/Spec/Prec/F1/AUROC

| 模型 | 类别 | Sens | Spec | Prec | F1 | AUROC |
|:--|:--|--:|--:|--:|--:|--:|
| ResNet34 | Benign lesions | 84.24±3.91 | 95.82±1.64 | 92.67±2.34 | 88.18±1.27 | 97.33±0.20 |
| ResNet34 | Malignant lesions | 91.27±2.79 | 92.12±2.17 | 84.21±3.42 | 87.53±1.17 | 97.25±0.06 |
| ResNet34 | body of stomach normal | 95.77±1.02 | 97.07±0.54 | 93.50±1.11 | 94.62±0.73 | 99.48±0.20 |
| FasterNet-T0 | Benign lesions | 86.59±2.60 | 93.73±1.97 | 89.61±2.63 | 88.02±0.05 | 96.41±0.38 |
| FasterNet-T0 | Malignant lesions | 86.83±3.64 | 93.93±1.35 | 86.78±2.16 | 86.74±0.79 | 96.83±0.36 |
| FasterNet-T0 | body of stomach normal | 97.07±0.49 | 97.00±0.21 | 93.43±0.46 | 95.22±0.41 | 99.65±0.07 |
| ViT-B/16 | Benign lesions | 84.90±7.56 | 93.82±3.25 | 89.80±4.40 | 87.02±2.70 | 96.77±0.25 |
| ViT-B/16 | Malignant lesions | 89.05±4.29 | 91.47±2.40 | 82.82±3.33 | 85.71±0.31 | 96.65±0.44 |
| ViT-B/16 | body of stomach normal | 92.36±3.32 | 97.35±1.46 | 94.01±2.89 | 93.11±0.23 | 99.12±0.32 |
| Swin-T | Benign lesions | 85.55±3.85 | 94.94±1.34 | 91.33±1.80 | 88.28±1.26 | 97.12±0.47 |
| Swin-T | Malignant lesions | 90.16±2.62 | 92.62±1.42 | 84.84±2.24 | 87.38±1.17 | 96.80±0.59 |
| Swin-T | body of stomach normal | 95.45±2.03 | 97.42±0.57 | 94.24±1.14 | 94.83±0.95 | 99.26±0.20 |
| PVT-Tiny | Benign lesions | 84.38±4.13 | 95.42±1.91 | 92.07±2.86 | 87.96±1.01 | 96.77±0.44 |
| PVT-Tiny | Malignant lesions | 91.43±2.52 | 92.05±1.60 | 84.05±2.31 | 87.54±0.57 | 96.72±0.23 |
| PVT-Tiny | body of stomach normal | 95.45±1.97 | 97.50±0.81 | 94.40±1.66 | 94.91±1.07 | 99.44±0.17 |
| RepViT-M1 | Benign lesions | 83.72±6.36 | 90.44±3.49 | 84.61±4.72 | 84.01±3.35 | 93.91±1.81 |
| RepViT-M1 | Malignant lesions | 79.05±6.55 | 94.07±2.23 | 86.06±4.34 | 82.25±3.57 | 94.73±1.14 |
| RepViT-M1 | body of stomach normal | 94.15±0.49 | 93.42±2.68 | 86.48±4.79 | 90.10±2.38 | 98.58±0.39 |
| **Ours** | Benign lesions | 86.33±3.41 | 95.90±0.64 | 92.88±0.88 | 89.45±1.61 | 97.15±0.75 |
| **Ours** | Malignant lesions | 90.95±1.72 | 92.91±2.71 | 85.61±4.58 | 88.13±1.67 | 97.24±0.50 |
| **Ours** | body of stomach normal | 96.59±1.76 | 97.57±0.66 | 94.61±1.32 | 95.57±0.52 | 99.64±0.03 |

## 表7　每类 HQ 相对 Full 提升（<span style="color:red">**红字，单位百分点**</span>）

Δ = HQ均值 − Full均值（仅基线有 Full 对照；Ours 为 “—”）。

| 模型 | 类别 | ΔSens | ΔSpec | ΔPrec | ΔF1 | ΔAUROC |
|:--|:--|--:|--:|--:|--:|--:|
| ResNet34 | Benign lesions | <span style="color:red">+1.56</span> | <span style="color:red">+2.81</span> | <span style="color:red">+4.63</span> | <span style="color:red">+2.95</span> | <span style="color:red">+2.19</span> |
| ResNet34 | Malignant lesions | <span style="color:red">+7.94</span> | <span style="color:red">+0.80</span> | <span style="color:red">+2.80</span> | <span style="color:red">+5.18</span> | <span style="color:red">+2.43</span> |
| ResNet34 | body of stomach normal | <span style="color:red">-1.95</span> | <span style="color:red">+0.29</span> | <span style="color:red">+0.43</span> | <span style="color:red">-0.70</span> | <span style="color:red">-0.11</span> |
| FasterNet-T0 | Benign lesions | <span style="color:red">-0.52</span> | <span style="color:red">+4.10</span> | <span style="color:red">+5.69</span> | <span style="color:red">+2.57</span> | <span style="color:red">+1.62</span> |
| FasterNet-T0 | Malignant lesions | <span style="color:red">+8.89</span> | <span style="color:red">-0.65</span> | <span style="color:red">-0.02</span> | <span style="color:red">+4.67</span> | <span style="color:red">+1.65</span> |
| FasterNet-T0 | body of stomach normal | <span style="color:red">-0.33</span> | <span style="color:red">+0.57</span> | <span style="color:red">+1.11</span> | <span style="color:red">+0.43</span> | <span style="color:red">+0.19</span> |
| ViT-B/16 | Benign lesions | <span style="color:red">+3.91</span> | <span style="color:red">+0.08</span> | <span style="color:red">+0.82</span> | <span style="color:red">+2.29</span> | <span style="color:red">+1.74</span> |
| ViT-B/16 | Malignant lesions | <span style="color:red">+0.95</span> | <span style="color:red">+3.33</span> | <span style="color:red">+5.53</span> | <span style="color:red">+3.44</span> | <span style="color:red">+1.67</span> |
| ViT-B/16 | body of stomach normal | <span style="color:red">+0.65</span> | <span style="color:red">-0.50</span> | <span style="color:red">-1.13</span> | <span style="color:red">-0.17</span> | <span style="color:red">+0.04</span> |
| Swin-T | Benign lesions | <span style="color:red">+2.73</span> | <span style="color:red">+2.17</span> | <span style="color:red">+3.15</span> | <span style="color:red">+3.14</span> | <span style="color:red">+1.33</span> |
| Swin-T | Malignant lesions | <span style="color:red">+5.71</span> | <span style="color:red">+1.16</span> | <span style="color:red">+2.36</span> | <span style="color:red">+4.30</span> | <span style="color:red">+0.78</span> |
| Swin-T | body of stomach normal | <span style="color:red">-0.49</span> | <span style="color:red">+0.79</span> | <span style="color:red">+1.39</span> | <span style="color:red">+0.55</span> | <span style="color:red">-0.13</span> |
| PVT-Tiny | Benign lesions | <span style="color:red">+0.91</span> | <span style="color:red">+3.45</span> | <span style="color:red">+4.83</span> | <span style="color:red">+2.97</span> | <span style="color:red">+1.04</span> |
| PVT-Tiny | Malignant lesions | <span style="color:red">+8.41</span> | <span style="color:red">+0.36</span> | <span style="color:red">+1.63</span> | <span style="color:red">+5.18</span> | <span style="color:red">+1.24</span> |
| PVT-Tiny | body of stomach normal | <span style="color:red">-0.65</span> | <span style="color:red">+0.57</span> | <span style="color:red">+1.17</span> | <span style="color:red">+0.27</span> | <span style="color:red">+0.16</span> |
| RepViT-M1 | Benign lesions | <span style="color:red">+6.51</span> | <span style="color:red">+5.62</span> | <span style="color:red">+8.71</span> | <span style="color:red">+7.54</span> | <span style="color:red">+4.06</span> |
| RepViT-M1 | Malignant lesions | <span style="color:red">+8.41</span> | <span style="color:red">+8.03</span> | <span style="color:red">+16.26</span> | <span style="color:red">+12.11</span> | <span style="color:red">+6.94</span> |
| RepViT-M1 | body of stomach normal | <span style="color:red">+8.13</span> | <span style="color:red">-2.00</span> | <span style="color:red">-2.99</span> | <span style="color:red">+2.61</span> | <span style="color:red">+1.02</span> |
| **Ours** | Benign lesions | — | — | — | — | — |
| **Ours** | Malignant lesions | — | — | — | — | — |
| **Ours** | body of stomach normal | — | — | — | — | — |

## 表8　模型复杂度与推理效率

- 复杂度与 seed 无关，取单次值；FPS 为原有 3-seed 均值（RTX 4090）。Ours 无 Full → Full 列 “—”；CUFIT 本次未测复杂度/推理效率，不从训练耗时推算 FPS

| 模型 | Params | MACs | FLOPs | Pure FPS(Full/HQ) | E2E FPS(Full/HQ) | Pure ms/图(Full/HQ) |
|:--|--:|--:|--:|--:|--:|--:|
| ResNet34 | 21.29M | 3.68G | 7.36G | 3797 / 3774 | 242 / 241 | 0.26 / 0.26 |
| FasterNet-T0 | 2.63M | 339.37M | 678.74M | 8969 / 9128 | 211 / 223 | 0.11 / 0.11 |
| ViT-B/16 | 85.65M | 16.86G | 33.73G | 869 / 869 | 200 / 199 | 1.15 / 1.15 |
| Swin-T | 27.94M | 4.46G | 8.92G | 1526 / 1525 | 219 / 233 | 0.66 / 0.66 |
| PVT-Tiny | 12.33M | 1.86G | 3.73G | 2911 / 2912 | 237 / 235 | 0.34 / 0.34 |
| RepViT-M1 | 6.41M | 1.14G | 2.28G | 3852 / 3857 | 218 / 213 | 0.26 / 0.26 |
| **Ours** | 21.95M | 3.87G | 7.74G | — / 976 | — / 201 | — / 1.02 |
| **CUFIT** | 未测 | 未测 | 未测 | 未测 / — | 未测 / — | 未测 / — |

## 表9　CUFIT Full 与三方共识完整方案对比（2026-10-22）

两条方案均从同一份 Full 原始训练集出发：CUFIT 直接训练 Full；本方法先用三方 OOF 共识筛出 HQ，再训练 Ours。seed 均为 42/43/44，共享相同 val/test，统计口径为均值±样本标准差。

| 方案 | 实际训练图像 | Macro-F1 | Accuracy | AUROC |
|:--|--:|--:|--:|--:|
| CUFIT + Full | 4977 | 90.43±0.38 | 90.31±0.39 | 97.97±0.03 |
| HQ + Ours | 2880 | 91.05±1.00 | 90.91±1.07 | 98.03±0.38 |
| RVC + ResNet34（消融参考） | 2880 | 90.11±0.98 | 89.97±1.01 | 98.02±0.15 |
| RC + ResNet34（消融参考） | 3141 | 91.24±1.15 | 91.11±1.14 | 97.97±0.19 |

**主比较：HQ + Ours 相对 CUFIT Full 的 Macro-F1、Accuracy、AUROC 分别高 0.619、0.596、0.063 个百分点。**这是三种子的描述性均值差异，未作显著性检验；不从单个最高种子得出方法优劣结论。

- RVC 为三方共识，RC 为双方共识，参考组不替代主比较。
- CUFIT 未补跑 HQ seed 43/44；历史 HQ seed 42 单次记录不混入三种子汇总，也不计算 CUFIT 的 HQ−Full 提升。
- 比较差异包含筛选、模型结构、预训练和训练/预处理策略，不能全部归因于三方共识；患者级划分本次未独立复核。
- CUFIT 保留 Resize(256×256)+CenterCrop(224)，Ours 保留短边 Resize(256)+CenterCrop(224)。
- seed 42 CUFIT 与三个 Ours 权重复测主要指标均与历史记录一致；旧 seed 42 的完整训练环境无法追溯。
- CUFIT/Ours 主比较使用本次高精度预测；RVC/RC 参考沿用历史 6 位小数预测，AUROC 可能有舍入差异。
- [完整报告](cufit_comparison_results/COMPARISON_REPORT.md) · [逐种子](cufit_comparison_results/PER_SEED.csv) · [总体](cufit_comparison_results/SUMMARY.csv) · [分类别](cufit_comparison_results/PER_CLASS.csv) · [CUFIT 补充逐种子指标](cufit_comparison_results/CUFIT_FULL_SEED_METRICS.csv)。

## 附录　3-seed 明细（Macro-F1 / Accuracy / AUROC，%）

| 模型 | 指标 | F·s42 | F·s43 | F·s44 | H·s42 | H·s43 | H·s44 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| ResNet34 | Macro-F1 | 87.81 | 87.73 | 87.37 | 90.44 | 89.01 | 90.89 |
|  | Accuracy | 87.63 | 87.63 | 87.18 | 90.31 | 88.82 | 90.76 |
|  | AUROC | 96.43 | 96.50 | 96.62 | 98.12 | 97.86 | 98.09 |
| FasterNet-T0 | Macro-F1 | 88.64 | 86.90 | 86.77 | 89.84 | 90.32 | 89.82 |
|  | Accuracy | 88.52 | 86.89 | 86.74 | 89.72 | 90.16 | 89.72 |
|  | AUROC | 96.93 | 96.21 | 96.29 | 97.67 | 97.89 | 97.39 |
| ViT-B/16 | Macro-F1 | 86.47 | 86.36 | 87.44 | 87.79 | 89.51 | 88.53 |
|  | Accuracy | 86.29 | 86.14 | 87.03 | 87.63 | 89.42 | 88.38 |
|  | AUROC | 95.75 | 95.99 | 97.35 | 97.71 | 97.59 | 97.23 |
| Swin-T | Macro-F1 | 87.54 | 87.84 | 87.13 | 89.59 | 91.08 | 89.82 |
|  | Accuracy | 87.18 | 87.78 | 87.03 | 89.42 | 90.91 | 89.72 |
|  | AUROC | 96.74 | 97.33 | 97.14 | 97.53 | 98.24 | 97.49 |
| PVT-Tiny | Macro-F1 | 87.54 | 85.81 | 88.64 | 89.58 | 90.50 | 90.32 |
|  | Accuracy | 87.33 | 85.69 | 88.52 | 89.42 | 90.31 | 90.16 |
|  | AUROC | 97.26 | 95.80 | 97.43 | 97.56 | 97.86 | 97.51 |
| RepViT-M1 | Macro-F1 | 78.35 | 77.60 | 78.16 | 86.65 | 82.06 | 87.65 |
|  | Accuracy | 77.94 | 77.65 | 77.94 | 86.59 | 82.12 | 87.63 |
|  | AUROC | 91.96 | 92.13 | 91.12 | 96.34 | 94.47 | 96.43 |
| **Ours** | Macro-F1 | — | — | — | 90.77 | 92.16 | 90.22 |
|  | Accuracy | — | — | — | 90.61 | 92.10 | 90.01 |
|  | AUROC | — | — | — | 97.60 | 98.34 | 98.16 |
| **CUFIT** | Macro-F1 | 90.87 | 90.23 | 90.19 | — | — | — |
|   | Accuracy | 90.76 | 90.16 | 90.01 | — | — | — |
|   | AUROC | 97.95 | 98.01 | 97.95 | — | — | — |

<!-- CFULL_LPM_COMPARISON_20261023 -->
## 新增实验：C_LPM 与完整 CUFIT C_Full（三方交集）

两臂均在同一配置下重新训练Ours，seed42/43/44；历史最高Ours仅作既有记录，不替代本次配对对照。

| 方案 | 训练N | Macro-F1 | Accuracy | AUROC |
|---|---:|---:|---:|---:|
| RVC_LPM → Ours | 2880 | 90.94±1.62 | 90.86±1.57 | 97.81±0.55 |
| RVC_FULL → Ours | 3011 | 89.70±0.72 | 89.57±0.79 | 97.87±0.14 |

完整结果与限制见 [C_Full vs C_LPM 对比报告](cfull_vs_lpm_results/COMPARISON_REPORT.md)。筛选器协议同时有内层验证留出变化；样本量/类别组成不同，结果仅作描述性比较，未进行显著性检验。

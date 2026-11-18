# starnet_s2_lp_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_lp_dce4s_seed42/starnet_s2_lp_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9660 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9981 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9738 |
| Balanced Acc | 0.9586 |
| Macro-Sensitivity | 0.9586 |
| Macro-Specificity | 0.9856 |
| Macro-Precision | 0.9748 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9588 |
| MCC | 0.9591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4033.1 (0.25 ms/张) |
| End-to-End FPS | 8.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9698 | 0.9650 | 0.9772 | 0.9912 | 0.9986 | 0.9983 | 195 |
| Malignant_lesions | 0.8861 | 0.9943 | 0.9722 | 0.9272 | 0.9746 | 0.9959 | 0.9842 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 0.9999 | 0.9998 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=7 TN=225 FN=2
- **Malignant_lesions**: TP=70 FP=2 TN=346 FN=9
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 7 | 70 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9650    0.9897    0.9772       195
     Malignant_lesions     0.9722    0.8861    0.9272        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9742       427
             macro avg     0.9748    0.9586    0.9660       427
          weighted avg     0.9743    0.9742    0.9738       427

```
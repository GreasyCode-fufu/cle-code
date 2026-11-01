# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9872 |
| Accuracy | 0.9883 |
| Macro-AUROC (OvR) | 0.9990 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9883 |
| Balanced Acc | 0.9864 |
| Macro-Sensitivity | 0.9864 |
| Macro-Specificity | 0.9937 |
| Macro-Precision | 0.9880 |
| AUC-PR (Macro) | 0.9971 |
| Cohen's Kappa | 0.9814 |
| MCC | 0.9814 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1668.9 (0.60 ms/张) |
| End-to-End FPS | 257.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9914 | 0.9897 | 0.9871 | 0.9871 | 0.9988 | 0.9986 | 195 |
| Malignant_lesions | 0.9747 | 0.9971 | 0.9872 | 0.9809 | 0.9943 | 0.9984 | 0.9929 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 0.9999 | 0.9998 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=2 TN=230 FN=3
- **Malignant_lesions**: TP=77 FP=1 TN=347 FN=2
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 1 | 2 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9897    0.9846    0.9871       195
     Malignant_lesions     0.9872    0.9747    0.9809        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9883       427
             macro avg     0.9880    0.9864    0.9872       427
          weighted avg     0.9883    0.9883    0.9883       427

```
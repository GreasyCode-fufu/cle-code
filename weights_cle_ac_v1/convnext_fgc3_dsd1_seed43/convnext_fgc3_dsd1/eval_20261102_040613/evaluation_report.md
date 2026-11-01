# convnext_fgc3_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd1_seed43/convnext_fgc3_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9677 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9739 |
| Balanced Acc | 0.9611 |
| Macro-Sensitivity | 0.9611 |
| Macro-Specificity | 0.9854 |
| Macro-Precision | 0.9754 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9588 |
| MCC | 0.9591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.39G |
| FLOPs | 8.78G |
| Params | 27.62M |
| Pure-Inference FPS | 1025.6 (0.98 ms/张) |
| End-to-End FPS | 231.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9655 | 0.9600 | 0.9722 | 0.9868 | 0.9972 | 0.9966 | 195 |
| Malignant_lesions | 0.8987 | 0.9943 | 0.9726 | 0.9342 | 0.9774 | 0.9931 | 0.9800 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=8 TN=224 FN=3
- **Malignant_lesions**: TP=71 FP=2 TN=346 FN=8
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9600    0.9846    0.9722       195
     Malignant_lesions     0.9726    0.8987    0.9342        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9754    0.9611    0.9677       427
          weighted avg     0.9743    0.9742    0.9739       427

```
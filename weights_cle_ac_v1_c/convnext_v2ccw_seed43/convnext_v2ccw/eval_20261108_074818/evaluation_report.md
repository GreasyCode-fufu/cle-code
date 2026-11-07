# convnext_v2ccw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2ccw_seed43/convnext_v2ccw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9834 |
| Accuracy | 0.9859 |
| Macro-AUROC (OvR) | 0.9989 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9872 |
| Macro-Sensitivity | 0.9872 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9798 |
| AUC-PR (Macro) | 0.9960 |
| Cohen's Kappa | 0.9777 |
| MCC | 0.9778 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 567.0 (1.76 ms/张) |
| End-to-End FPS | 197.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9957 | 0.9948 | 0.9845 | 0.9788 | 0.9988 | 0.9987 | 195 |
| Malignant_lesions | 0.9873 | 0.9885 | 0.9512 | 0.9689 | 0.9971 | 0.9978 | 0.9893 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=1 TN=231 FN=5
- **Malignant_lesions**: TP=78 FP=4 TN=344 FN=1
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 1 | 78 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9948    0.9744    0.9845       195
     Malignant_lesions     0.9512    0.9873    0.9689        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9859       427
             macro avg     0.9798    0.9872    0.9834       427
          weighted avg     0.9863    0.9859    0.9860       427

```
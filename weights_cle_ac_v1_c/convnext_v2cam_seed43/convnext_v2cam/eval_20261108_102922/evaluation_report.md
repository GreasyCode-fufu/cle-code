# convnext_v2cam 评估报告

- 权重: `weights_cle_ac_v1_c/convnext_v2cam_seed43/convnext_v2cam/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9774 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9813 |
| Balanced Acc | 0.9788 |
| Macro-Sensitivity | 0.9788 |
| Macro-Specificity | 0.9906 |
| Macro-Precision | 0.9760 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9702 |
| MCC | 0.9703 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 387.4 (2.58 ms/张) |
| End-to-End FPS | 16.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9871 | 0.9845 | 0.9794 | 0.9786 | 0.9982 | 0.9979 | 195 |
| Malignant_lesions | 0.9620 | 0.9885 | 0.9500 | 0.9560 | 0.9914 | 0.9964 | 0.9838 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=3 TN=229 FN=5
- **Malignant_lesions**: TP=76 FP=4 TN=344 FN=3
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9744    0.9794       195
     Malignant_lesions     0.9500    0.9620    0.9560        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9760    0.9788    0.9774       427
          weighted avg     0.9813    0.9813    0.9813       427

```
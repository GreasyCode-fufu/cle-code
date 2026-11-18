# convnext_fgc3_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd2_seed43/convnext_fgc3_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9771 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9978 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9813 |
| Balanced Acc | 0.9783 |
| Macro-Sensitivity | 0.9783 |
| Macro-Specificity | 0.9906 |
| Macro-Precision | 0.9760 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9702 |
| MCC | 0.9702 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.41G |
| FLOPs | 8.81G |
| Params | 27.61M |
| Pure-Inference FPS | 351.2 (2.85 ms/张) |
| End-to-End FPS | 162.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9871 | 0.9845 | 0.9820 | 0.9828 | 0.9975 | 0.9970 | 195 |
| Malignant_lesions | 0.9620 | 0.9885 | 0.9500 | 0.9560 | 0.9914 | 0.9963 | 0.9844 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9964 | 0.9935 | 0.9935 | 0.9964 | 0.9997 | 0.9995 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=3 TN=229 FN=4
- **Malignant_lesions**: TP=76 FP=4 TN=344 FN=3
- **body_of_stomach_normal**: TP=152 FP=1 TN=273 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 2 | 76 | 1 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9795    0.9820       195
     Malignant_lesions     0.9500    0.9620    0.9560        79
body_of_stomach_normal     0.9935    0.9935    0.9935       153

              accuracy                         0.9813       427
             macro avg     0.9760    0.9783    0.9771       427
          weighted avg     0.9813    0.9813    0.9813       427

```
# convnext_fesr 评估报告

- 权重: `weights_cle_ac_v1/convnext_fesr_seed42/convnext_fesr/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9701 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9766 |
| Balanced Acc | 0.9699 |
| Macro-Sensitivity | 0.9699 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9704 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9627 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1305.3 (0.77 ms/张) |
| End-to-End FPS | 199.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9784 | 0.9745 | 0.9770 | 0.9827 | 0.9981 | 0.9978 | 195 |
| Malignant_lesions | 0.9367 | 0.9856 | 0.9367 | 0.9367 | 0.9856 | 0.9960 | 0.9823 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=5 TN=227 FN=4
- **Malignant_lesions**: TP=74 FP=5 TN=343 FN=5
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9745    0.9795    0.9770       195
     Malignant_lesions     0.9367    0.9367    0.9367        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9766       427
             macro avg     0.9704    0.9699    0.9701       427
          weighted avg     0.9766    0.9766    0.9766       427

```
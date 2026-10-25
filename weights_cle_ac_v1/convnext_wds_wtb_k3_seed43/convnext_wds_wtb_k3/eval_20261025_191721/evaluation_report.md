# convnext_wds_wtb_k3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_seed43/convnext_wds_wtb_k3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9461 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9462 |
| Balanced Acc | 0.9479 |
| Macro-Sensitivity | 0.9479 |
| Macro-Specificity | 0.9735 |
| Macro-Precision | 0.9458 |
| AUC-PR (Macro) | 0.9894 |
| Cohen's Kappa | 0.9192 |
| MCC | 0.9199 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.50G |
| FLOPs | 9.01G |
| Params | 28.28M |
| Pure-Inference FPS | 777.7 (1.29 ms/张) |
| End-to-End FPS | 217.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9783 | 0.9590 | 0.9280 | 0.9449 | 0.9923 | 0.9872 | 208 |
| Malignant_lesions | 0.9602 | 0.9500 | 0.8942 | 0.9260 | 0.9819 | 0.9921 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=8 TN=360 FN=21
- **Malignant_lesions**: TP=169 FP=20 TN=380 FN=7
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 20 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9590    0.8990    0.9280       208
     Malignant_lesions     0.8942    0.9602    0.9260       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9462       576
             macro avg     0.9458    0.9479    0.9461       576
          weighted avg     0.9476    0.9462    0.9462       576

```
# convnext_v2ab 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2ab_seed42/convnext_v2ab/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9494 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9480 |
| Macro-Sensitivity | 0.9480 |
| Macro-Specificity | 0.9744 |
| Macro-Precision | 0.9517 |
| AUC-PR (Macro) | 0.9850 |
| Cohen's Kappa | 0.9242 |
| MCC | 0.9247 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.97G |
| Params | 28.22M |
| Pure-Inference FPS | 1489.2 (0.67 ms/张) |
| End-to-End FPS | 261.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9511 | 0.9174 | 0.9390 | 0.9777 | 0.9898 | 0.9824 | 208 |
| Malignant_lesions | 0.9034 | 0.9850 | 0.9636 | 0.9326 | 0.9586 | 0.9866 | 0.9748 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=18 TN=350 FN=8
- **Malignant_lesions**: TP=159 FP=6 TN=394 FN=17
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9174    0.9615    0.9390       208
     Malignant_lesions     0.9636    0.9034    0.9326       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9497       576
             macro avg     0.9517    0.9480    0.9494       576
          weighted avg     0.9504    0.9497    0.9496       576

```
# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9479 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9741 |
| Macro-Precision | 0.9475 |
| AUC-PR (Macro) | 0.9894 |
| Cohen's Kappa | 0.9218 |
| MCC | 0.9220 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 972.5 (1.03 ms/张) |
| End-to-End FPS | 59.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9701 | 0.9455 | 0.9317 | 0.9545 | 0.9922 | 0.9869 | 208 |
| Malignant_lesions | 0.9489 | 0.9575 | 0.9076 | 0.9278 | 0.9770 | 0.9915 | 0.9822 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=11 TN=357 FN=17
- **Malignant_lesions**: TP=167 FP=17 TN=383 FN=9
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 16 | 1 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9455    0.9183    0.9317       208
     Malignant_lesions     0.9076    0.9489    0.9278       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9479       576
             macro avg     0.9475    0.9488    0.9479       576
          weighted avg     0.9486    0.9479    0.9480       576

```
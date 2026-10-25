# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9471 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9931 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9475 |
| Balanced Acc | 0.9470 |
| Macro-Sensitivity | 0.9470 |
| Macro-Specificity | 0.9738 |
| Macro-Precision | 0.9477 |
| AUC-PR (Macro) | 0.9877 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9219 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 343.3 (2.91 ms/张) |
| End-to-End FPS | 160.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9647 | 0.9375 | 0.9375 | 0.9647 | 0.9889 | 0.9831 | 208 |
| Malignant_lesions | 0.9034 | 0.9750 | 0.9408 | 0.9217 | 0.9582 | 0.9907 | 0.9808 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9818 | 0.9648 | 0.9821 | 1.0000 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=13 TN=355 FN=13
- **Malignant_lesions**: TP=159 FP=10 TN=390 FN=17
- **body_of_stomach_normal**: TP=192 FP=7 TN=377 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9375    0.9375    0.9375       208
     Malignant_lesions     0.9408    0.9034    0.9217       176
body_of_stomach_normal     0.9648    1.0000    0.9821       192

              accuracy                         0.9479       576
             macro avg     0.9477    0.9470    0.9471       576
          weighted avg     0.9476    0.9479    0.9475       576

```
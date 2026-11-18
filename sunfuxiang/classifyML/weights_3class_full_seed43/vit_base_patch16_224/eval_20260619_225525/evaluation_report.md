# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed43/vit_base_patch16_224/vit_base_patch16_224_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8636 |
| Accuracy | 0.8614 |
| Macro-AUROC (OvR) | 0.9599 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8627 |
| Balanced Acc | 0.8622 |
| Macro-Sensitivity | 0.8622 |
| Macro-Specificity | 0.9300 |
| Macro-Precision | 0.8675 |
| AUC-PR (Macro) | 0.9252 |
| Cohen's Kappa | 0.7910 |
| MCC | 0.7921 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 867.9 (1.15 ms/张) |
| End-to-End FPS | 211.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9157 | 0.8617 | 0.8566 | 0.9091 | 0.9462 | 0.9315 | 256 |
| Malignant_lesions | 0.8619 | 0.8915 | 0.7835 | 0.8209 | 0.9341 | 0.9452 | 0.8701 | 210 |
| body_of_stomach_normal | 0.8732 | 0.9828 | 0.9572 | 0.9133 | 0.9463 | 0.9884 | 0.9741 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=35 TN=380 FN=38
- **Malignant_lesions**: TP=181 FP=50 TN=411 FN=29
- **body_of_stomach_normal**: TP=179 FP=8 TN=458 FN=26

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 32 | 6 |
| Malignant_lesions | 27 | 181 | 2 |
| body_of_stomach_normal | 8 | 18 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8617    0.8516    0.8566       256
     Malignant_lesions     0.7835    0.8619    0.8209       210
body_of_stomach_normal     0.9572    0.8732    0.9133       205

              accuracy                         0.8614       671
             macro avg     0.8675    0.8622    0.8636       671
          weighted avg     0.8664    0.8614    0.8627       671

```
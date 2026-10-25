# convnext_wtb_k3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wtb_k3_seed42/convnext_wtb_k3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9451 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9943 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9458 |
| Balanced Acc | 0.9454 |
| Macro-Sensitivity | 0.9454 |
| Macro-Specificity | 0.9731 |
| Macro-Precision | 0.9452 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9191 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 28.27M |
| Pure-Inference FPS | 781.4 (1.28 ms/张) |
| End-to-End FPS | 221.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9701 | 0.9463 | 0.9395 | 0.9623 | 0.9931 | 0.9886 | 208 |
| Malignant_lesions | 0.9034 | 0.9675 | 0.9244 | 0.9138 | 0.9579 | 0.9902 | 0.9782 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9818 | 0.9648 | 0.9821 | 1.0000 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=11 TN=357 FN=14
- **Malignant_lesions**: TP=159 FP=13 TN=387 FN=17
- **body_of_stomach_normal**: TP=192 FP=7 TN=377 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 11 | 159 | 6 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9463    0.9327    0.9395       208
     Malignant_lesions     0.9244    0.9034    0.9138       176
body_of_stomach_normal     0.9648    1.0000    0.9821       192

              accuracy                         0.9462       576
             macro avg     0.9452    0.9454    0.9451       576
          weighted avg     0.9458    0.9462    0.9458       576

```
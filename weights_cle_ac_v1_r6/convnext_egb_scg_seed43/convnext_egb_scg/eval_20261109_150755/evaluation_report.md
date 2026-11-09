# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9546 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9556 |
| Macro-Sensitivity | 0.9556 |
| Macro-Specificity | 0.9777 |
| Macro-Precision | 0.9541 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9325 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 334.0 (2.99 ms/张) |
| End-to-End FPS | 161.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9837 | 0.9698 | 0.9484 | 0.9602 | 0.9932 | 0.9867 | 208 |
| Malignant_lesions | 0.9545 | 0.9650 | 0.9231 | 0.9385 | 0.9797 | 0.9912 | 0.9851 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=6 TN=362 FN=15
- **Malignant_lesions**: TP=168 FP=14 TN=386 FN=8
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 12 | 3 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9698    0.9279    0.9484       208
     Malignant_lesions     0.9231    0.9545    0.9385       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9549       576
             macro avg     0.9541    0.9556    0.9546       576
          weighted avg     0.9554    0.9549    0.9548       576

```
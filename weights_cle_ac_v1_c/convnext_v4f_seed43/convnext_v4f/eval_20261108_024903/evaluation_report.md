# convnext_v4f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4f_seed43/convnext_v4f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9443 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9445 |
| Balanced Acc | 0.9430 |
| Macro-Sensitivity | 0.9430 |
| Macro-Specificity | 0.9718 |
| Macro-Precision | 0.9464 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9168 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 861.8 (1.16 ms/张) |
| End-to-End FPS | 229.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9484 | 0.9136 | 0.9393 | 0.9803 | 0.9941 | 0.9910 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9933 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9896 | 0.9785 | 0.9630 | 0.9744 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=19 TN=349 FN=7
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=182 FP=4 TN=380 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 6 | 4 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9136    0.9663    0.9393       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9785    0.9479    0.9630       192

              accuracy                         0.9444       576
             macro avg     0.9464    0.9430    0.9443       576
          weighted avg     0.9455    0.9444    0.9445       576

```
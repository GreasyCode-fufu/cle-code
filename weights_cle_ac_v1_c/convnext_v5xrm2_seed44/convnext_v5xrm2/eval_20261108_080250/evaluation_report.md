# convnext_v5xrm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrm2_seed44/convnext_v5xrm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9651 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9652 |
| Balanced Acc | 0.9664 |
| Macro-Sensitivity | 0.9664 |
| Macro-Specificity | 0.9829 |
| Macro-Precision | 0.9646 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9479 |
| MCC | 0.9482 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 461.2 (2.17 ms/张) |
| End-to-End FPS | 184.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9891 | 0.9798 | 0.9557 | 0.9630 | 0.9946 | 0.9929 | 208 |
| Malignant_lesions | 0.9716 | 0.9700 | 0.9344 | 0.9526 | 0.9873 | 0.9967 | 0.9923 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=4 TN=364 FN=14
- **Malignant_lesions**: TP=171 FP=12 TN=388 FN=5
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 3 | 171 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9798    0.9327    0.9557       208
     Malignant_lesions     0.9344    0.9716    0.9526       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9653       576
             macro avg     0.9646    0.9664    0.9651       576
          weighted avg     0.9658    0.9653    0.9652       576

```
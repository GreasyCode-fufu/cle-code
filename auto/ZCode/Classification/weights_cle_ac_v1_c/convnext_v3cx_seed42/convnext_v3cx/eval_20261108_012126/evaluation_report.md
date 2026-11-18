# convnext_v3cx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cx_seed42/convnext_v3cx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9686 |
| Accuracy | 0.9688 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9687 |
| Balanced Acc | 0.9675 |
| Macro-Sensitivity | 0.9675 |
| Macro-Specificity | 0.9840 |
| Macro-Precision | 0.9703 |
| AUC-PR (Macro) | 0.9949 |
| Cohen's Kappa | 0.9530 |
| MCC | 0.9532 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 617.3 (1.62 ms/张) |
| End-to-End FPS | 205.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9674 | 0.9444 | 0.9623 | 0.9889 | 0.9964 | 0.9941 | 208 |
| Malignant_lesions | 0.9375 | 0.9925 | 0.9821 | 0.9593 | 0.9730 | 0.9965 | 0.9918 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9994 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=12 TN=356 FN=4
- **Malignant_lesions**: TP=165 FP=3 TN=397 FN=11
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 3 | 1 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9444    0.9808    0.9623       208
     Malignant_lesions     0.9821    0.9375    0.9593       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9688       576
             macro avg     0.9703    0.9675    0.9686       576
          weighted avg     0.9693    0.9688    0.9687       576

```
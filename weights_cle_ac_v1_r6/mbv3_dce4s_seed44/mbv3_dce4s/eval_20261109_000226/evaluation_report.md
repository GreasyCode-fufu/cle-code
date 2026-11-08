# mbv3_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4s_seed44/mbv3_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9517 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9515 |
| Balanced Acc | 0.9513 |
| Macro-Sensitivity | 0.9513 |
| Macro-Specificity | 0.9754 |
| Macro-Precision | 0.9523 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9269 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2923.0 (0.34 ms/张) |
| End-to-End FPS | 11.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9565 | 0.9245 | 0.9333 | 0.9670 | 0.9876 | 0.9796 | 208 |
| Malignant_lesions | 0.9375 | 0.9775 | 0.9483 | 0.9429 | 0.9726 | 0.9900 | 0.9762 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9974 | 0.9954 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=16 TN=352 FN=12
- **Malignant_lesions**: TP=165 FP=9 TN=391 FN=11
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 9 | 3 |
| Malignant_lesions | 11 | 165 | 0 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9245    0.9423    0.9333       208
     Malignant_lesions     0.9483    0.9375    0.9429       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9514       576
             macro avg     0.9523    0.9513    0.9517       576
          weighted avg     0.9517    0.9514    0.9515       576

```
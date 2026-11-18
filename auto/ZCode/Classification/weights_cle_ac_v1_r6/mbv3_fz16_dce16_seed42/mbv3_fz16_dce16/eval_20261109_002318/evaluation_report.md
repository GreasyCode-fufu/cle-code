# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9654 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9658 |
| Macro-Sensitivity | 0.9658 |
| Macro-Specificity | 0.9826 |
| Macro-Precision | 0.9650 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9479 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5383.0 (0.19 ms/张) |
| End-to-End FPS | 14.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9783 | 0.9610 | 0.9540 | 0.9704 | 0.9930 | 0.9904 | 208 |
| Malignant_lesions | 0.9659 | 0.9775 | 0.9497 | 0.9577 | 0.9849 | 0.9951 | 0.9863 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9991 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=8 TN=360 FN=11
- **Malignant_lesions**: TP=170 FP=9 TN=391 FN=6
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9610    0.9471    0.9540       208
     Malignant_lesions     0.9497    0.9659    0.9577       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9653       576
             macro avg     0.9650    0.9658    0.9654       576
          weighted avg     0.9653    0.9653    0.9653       576

```
# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed45/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9358 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9359 |
| Balanced Acc | 0.9354 |
| Macro-Sensitivity | 0.9354 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.9364 |
| AUC-PR (Macro) | 0.9836 |
| Cohen's Kappa | 0.9034 |
| MCC | 0.9034 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4177.8 (0.24 ms/张) |
| End-to-End FPS | 13.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9511 | 0.9151 | 0.9238 | 0.9615 | 0.9860 | 0.9775 | 208 |
| Malignant_lesions | 0.9205 | 0.9675 | 0.9257 | 0.9231 | 0.9651 | 0.9883 | 0.9755 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9844 | 0.9683 | 0.9606 | 0.9767 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=18 TN=350 FN=14
- **Malignant_lesions**: TP=162 FP=13 TN=387 FN=14
- **body_of_stomach_normal**: TP=183 FP=6 TN=378 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 9 | 5 |
| Malignant_lesions | 13 | 162 | 1 |
| body_of_stomach_normal | 5 | 4 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9151    0.9327    0.9238       208
     Malignant_lesions     0.9257    0.9205    0.9231       176
body_of_stomach_normal     0.9683    0.9531    0.9606       192

              accuracy                         0.9358       576
             macro avg     0.9364    0.9354    0.9358       576
          weighted avg     0.9361    0.9358    0.9359       576

```
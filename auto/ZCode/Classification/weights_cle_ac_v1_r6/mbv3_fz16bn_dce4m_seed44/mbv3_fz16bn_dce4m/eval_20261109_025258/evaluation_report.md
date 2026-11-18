# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed44/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9353 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9356 |
| Balanced Acc | 0.9335 |
| Macro-Sensitivity | 0.9335 |
| Macro-Specificity | 0.9672 |
| Macro-Precision | 0.9391 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.9032 |
| MCC | 0.9041 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4245.6 (0.24 ms/张) |
| End-to-End FPS | 13.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9321 | 0.8879 | 0.9188 | 0.9717 | 0.9877 | 0.9804 | 208 |
| Malignant_lesions | 0.8693 | 0.9800 | 0.9503 | 0.9080 | 0.9446 | 0.9899 | 0.9769 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=25 TN=343 FN=10
- **Malignant_lesions**: TP=153 FP=8 TN=392 FN=23
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 21 | 153 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8879    0.9519    0.9188       208
     Malignant_lesions     0.9503    0.8693    0.9080       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9358       576
             macro avg     0.9391    0.9335    0.9353       576
          weighted avg     0.9374    0.9358    0.9356       576

```
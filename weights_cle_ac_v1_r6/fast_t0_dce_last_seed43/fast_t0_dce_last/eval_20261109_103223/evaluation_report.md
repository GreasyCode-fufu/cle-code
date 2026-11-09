# fast_t0_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce_last_seed43/fast_t0_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9407 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9919 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9409 |
| Balanced Acc | 0.9394 |
| Macro-Sensitivity | 0.9394 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9429 |
| AUC-PR (Macro) | 0.9841 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9115 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7742.1 (0.13 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9880 | 0.9804 | 208 |
| Malignant_lesions | 0.8920 | 0.9775 | 0.9458 | 0.9181 | 0.9537 | 0.9885 | 0.9733 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=157 FP=9 TN=391 FN=19
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 17 | 157 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9458    0.8920    0.9181       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9410       576
             macro avg     0.9429    0.9394    0.9407       576
          weighted avg     0.9417    0.9410    0.9409       576

```
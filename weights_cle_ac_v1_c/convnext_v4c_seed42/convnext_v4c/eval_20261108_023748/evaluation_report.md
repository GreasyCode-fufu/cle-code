# convnext_v4c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4c_seed42/convnext_v4c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9415 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9927 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9412 |
| Balanced Acc | 0.9402 |
| Macro-Sensitivity | 0.9402 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9433 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9115 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.85G |
| FLOPs | 3.71G |
| Params | 3.45M |
| Pure-Inference FPS | 409.8 (2.44 ms/张) |
| End-to-End FPS | 180.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9869 | 0.9800 | 208 |
| Malignant_lesions | 0.9205 | 0.9800 | 0.9529 | 0.9364 | 0.9655 | 0.9941 | 0.9874 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9870 | 0.9734 | 0.9632 | 0.9768 | 0.9973 | 0.9944 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=162 FP=8 TN=392 FN=14
- **body_of_stomach_normal**: TP=183 FP=5 TN=379 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 6 | 5 |
| Malignant_lesions | 14 | 162 | 0 |
| body_of_stomach_normal | 7 | 2 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9529    0.9205    0.9364       176
body_of_stomach_normal     0.9734    0.9531    0.9632       192

              accuracy                         0.9410       576
             macro avg     0.9433    0.9402    0.9415       576
          weighted avg     0.9420    0.9410    0.9412       576

```
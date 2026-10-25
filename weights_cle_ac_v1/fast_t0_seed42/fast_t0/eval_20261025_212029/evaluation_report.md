# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9616 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9696 |
| Balanced Acc | 0.9627 |
| Macro-Sensitivity | 0.9627 |
| Macro-Specificity | 0.9847 |
| Macro-Precision | 0.9605 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9516 |
| MCC | 0.9516 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 3428.6 (0.29 ms/张) |
| End-to-End FPS | 259.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9741 | 0.9691 | 0.9666 | 0.9700 | 0.9964 | 0.9959 | 195 |
| Malignant_lesions | 0.9241 | 0.9799 | 0.9125 | 0.9182 | 0.9827 | 0.9940 | 0.9748 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=6 TN=226 FN=7
- **Malignant_lesions**: TP=73 FP=7 TN=341 FN=6
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9691    0.9641    0.9666       195
     Malignant_lesions     0.9125    0.9241    0.9182        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9696       427
             macro avg     0.9605    0.9627    0.9616       427
          weighted avg     0.9697    0.9696    0.9696       427

```
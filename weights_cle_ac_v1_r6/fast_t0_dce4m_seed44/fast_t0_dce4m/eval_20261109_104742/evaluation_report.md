# fast_t0_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce4m_seed44/fast_t0_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9385 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9389 |
| Balanced Acc | 0.9368 |
| Macro-Sensitivity | 0.9368 |
| Macro-Specificity | 0.9690 |
| Macro-Precision | 0.9423 |
| AUC-PR (Macro) | 0.9854 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9094 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8759.4 (0.11 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9375 | 0.8964 | 0.9256 | 0.9746 | 0.9882 | 0.9803 | 208 |
| Malignant_lesions | 0.8693 | 0.9825 | 0.9563 | 0.9107 | 0.9447 | 0.9894 | 0.9780 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=23 TN=345 FN=9
- **Malignant_lesions**: TP=153 FP=7 TN=393 FN=23
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 20 | 153 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8964    0.9567    0.9256       208
     Malignant_lesions     0.9563    0.8693    0.9107       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9392       576
             macro avg     0.9423    0.9368    0.9385       576
          weighted avg     0.9406    0.9392    0.9389       576

```
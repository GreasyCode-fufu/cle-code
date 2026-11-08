# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shufflenet_v2_x1_0_seed45/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9218 |
| Accuracy | 0.9219 |
| Macro-AUROC (OvR) | 0.9868 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9220 |
| Balanced Acc | 0.9188 |
| Macro-Sensitivity | 0.9188 |
| Macro-Specificity | 0.9599 |
| Macro-Precision | 0.9288 |
| AUC-PR (Macro) | 0.9786 |
| Cohen's Kappa | 0.8822 |
| MCC | 0.8842 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 5280.9 (0.19 ms/张) |
| End-to-End FPS | 242.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9103 | 0.8578 | 0.9045 | 0.9738 | 0.9840 | 0.9734 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9864 | 0.9742 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9870 | 0.9734 | 0.9632 | 0.9768 | 0.9899 | 0.9882 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=33 TN=335 FN=9
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=183 FP=5 TN=379 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 24 | 149 | 3 |
| body_of_stomach_normal | 9 | 0 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8578    0.9567    0.9045       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9734    0.9531    0.9632       192

              accuracy                         0.9219       576
             macro avg     0.9288    0.9188    0.9218       576
          weighted avg     0.9261    0.9219    0.9220       576

```
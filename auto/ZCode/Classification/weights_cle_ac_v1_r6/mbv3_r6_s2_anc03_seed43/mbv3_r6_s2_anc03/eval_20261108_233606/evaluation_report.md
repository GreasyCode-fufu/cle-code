# mbv3_r6_s2_anc03 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_anc03_seed43/mbv3_r6_s2_anc03/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9296 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9301 |
| Balanced Acc | 0.9276 |
| Macro-Sensitivity | 0.9276 |
| Macro-Specificity | 0.9645 |
| Macro-Precision | 0.9348 |
| AUC-PR (Macro) | 0.9817 |
| Cohen's Kappa | 0.8953 |
| MCC | 0.8967 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4557.6 (0.22 ms/张) |
| End-to-End FPS | 255.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9266 | 0.8800 | 0.9145 | 0.9715 | 0.9861 | 0.9750 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9853 | 0.9716 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=27 TN=341 FN=10
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 24 | 149 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8800    0.9519    0.9145       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9306       576
             macro avg     0.9348    0.9276    0.9296       576
          weighted avg     0.9327    0.9306    0.9301       576

```
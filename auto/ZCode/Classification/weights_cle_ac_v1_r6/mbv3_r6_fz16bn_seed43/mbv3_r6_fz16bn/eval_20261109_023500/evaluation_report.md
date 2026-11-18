# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed43/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9233 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9235 |
| Balanced Acc | 0.9207 |
| Macro-Sensitivity | 0.9207 |
| Macro-Specificity | 0.9609 |
| Macro-Precision | 0.9294 |
| AUC-PR (Macro) | 0.9846 |
| Cohen's Kappa | 0.8848 |
| MCC | 0.8865 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4169.1 (0.24 ms/张) |
| End-to-End FPS | 101.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9158 | 0.8646 | 0.9062 | 0.9712 | 0.9881 | 0.9805 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9900 | 0.9785 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9844 | 0.9686 | 0.9661 | 0.9818 | 0.9972 | 0.9949 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=31 TN=337 FN=10
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=185 FP=6 TN=378 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 24 | 149 | 3 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8646    0.9519    0.9062       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9686    0.9635    0.9661       192

              accuracy                         0.9236       576
             macro avg     0.9294    0.9207    0.9233       576
          weighted avg     0.9269    0.9236    0.9235       576

```
# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tmp/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9487 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9492 |
| Balanced Acc | 0.9468 |
| Macro-Sensitivity | 0.9468 |
| Macro-Specificity | 0.9742 |
| Macro-Precision | 0.9531 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9241 |
| MCC | 0.9253 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 674.6 (1.48 ms/张) |
| End-to-End FPS | 211.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9457 | 0.9103 | 0.9420 | 0.9858 | 0.9945 | 0.9916 | 208 |
| Malignant_lesions | 0.8750 | 0.9900 | 0.9747 | 0.9222 | 0.9474 | 0.9959 | 0.9899 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9992 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=20 TN=348 FN=5
- **Malignant_lesions**: TP=154 FP=4 TN=396 FN=22
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 18 | 154 | 4 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9103    0.9760    0.9420       208
     Malignant_lesions     0.9747    0.8750    0.9222       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9497       576
             macro avg     0.9531    0.9468    0.9487       576
          weighted avg     0.9513    0.9497    0.9492       576

```
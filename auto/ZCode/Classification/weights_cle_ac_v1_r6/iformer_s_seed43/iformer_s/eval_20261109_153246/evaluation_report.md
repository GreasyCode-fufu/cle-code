# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_s_seed43/iformer_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9293 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9942 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9299 |
| Balanced Acc | 0.9270 |
| Macro-Sensitivity | 0.9270 |
| Macro-Specificity | 0.9644 |
| Macro-Precision | 0.9362 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.8953 |
| MCC | 0.8973 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2881.5 (0.35 ms/张) |
| End-to-End FPS | 215.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9239 | 0.8772 | 0.9174 | 0.9770 | 0.9915 | 0.9869 | 208 |
| Malignant_lesions | 0.8352 | 0.9875 | 0.9671 | 0.8963 | 0.9316 | 0.9925 | 0.9820 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=28 TN=340 FN=8
- **Malignant_lesions**: TP=147 FP=5 TN=395 FN=29
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 25 | 147 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8772    0.9615    0.9174       208
     Malignant_lesions     0.9671    0.8352    0.8963       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9306       576
             macro avg     0.9362    0.9270    0.9293       576
          weighted avg     0.9337    0.9306    0.9299       576

```
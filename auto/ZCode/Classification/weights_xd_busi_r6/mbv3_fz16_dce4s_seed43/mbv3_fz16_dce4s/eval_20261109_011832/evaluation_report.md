# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8339 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9477 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8473 |
| Balanced Acc | 0.8367 |
| Macro-Sensitivity | 0.8367 |
| Macro-Specificity | 0.9119 |
| Macro-Precision | 0.8316 |
| AUC-PR (Macro) | 0.9015 |
| Cohen's Kappa | 0.7396 |
| MCC | 0.7397 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1739.8 (0.57 ms/张) |
| End-to-End FPS | 163.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9382 | 0.9562 | 66 |
| malignant | 0.7812 | 0.9302 | 0.8065 | 0.7937 | 0.9195 | 0.9524 | 0.8758 | 32 |
| normal | 0.8500 | 0.9592 | 0.8095 | 0.8293 | 0.9691 | 0.9526 | 0.8724 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=25 FP=6 TN=80 FN=7
- **normal**: TP=17 FP=4 TN=94 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 6 | 2 |
| malignant | 5 | 25 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.8065    0.7812    0.7937        32
      normal     0.8095    0.8500    0.8293        20

    accuracy                         0.8475       118
   macro avg     0.8316    0.8367    0.8339       118
weighted avg     0.8474    0.8475    0.8473       118

```
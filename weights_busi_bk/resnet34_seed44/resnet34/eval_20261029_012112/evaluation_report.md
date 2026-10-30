# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed44/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8833 |
| Accuracy | 0.9052 |
| Macro-AUROC (OvR) | 0.9295 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9049 |
| Balanced Acc | 0.8809 |
| Macro-Sensitivity | 0.8809 |
| Macro-Specificity | 0.9456 |
| Macro-Precision | 0.8878 |
| AUC-PR (Macro) | 0.8530 |
| Cohen's Kappa | 0.8367 |
| MCC | 0.8375 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1931.9 (0.52 ms/张) |
| End-to-End FPS | 177.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9020 | 0.9254 | 0.9394 | 0.9388 | 0.9532 | 0.9504 | 65 |
| malignant | 0.8387 | 0.9765 | 0.9286 | 0.8814 | 0.9432 | 0.9427 | 0.7938 | 31 |
| normal | 0.8500 | 0.9583 | 0.8095 | 0.8293 | 0.9684 | 0.8927 | 0.8148 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=5 TN=46 FN=3
- **malignant**: TP=26 FP=2 TN=83 FN=5
- **normal**: TP=17 FP=4 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 1 | 2 |
| malignant | 3 | 26 | 2 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9254    0.9538    0.9394        65
   malignant     0.9286    0.8387    0.8814        31
      normal     0.8095    0.8500    0.8293        20

    accuracy                         0.9052       116
   macro avg     0.8878    0.8809    0.8833       116
weighted avg     0.9063    0.9052    0.9049       116

```
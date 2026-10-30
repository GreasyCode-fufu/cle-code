# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed44/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8090 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9428 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8210 |
| Balanced Acc | 0.8108 |
| Macro-Sensitivity | 0.8108 |
| Macro-Specificity | 0.8931 |
| Macro-Precision | 0.8108 |
| AUC-PR (Macro) | 0.9056 |
| Cohen's Kappa | 0.6941 |
| MCC | 0.6951 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1975.6 (0.51 ms/张) |
| End-to-End FPS | 185.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.7885 | 0.8382 | 0.8507 | 0.8200 | 0.9277 | 0.9461 | 66 |
| malignant | 0.7188 | 0.9419 | 0.8214 | 0.7667 | 0.9000 | 0.9248 | 0.8813 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9760 | 0.8893 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=11 TN=41 FN=9
- **malignant**: TP=23 FP=5 TN=81 FN=9
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 5 | 4 |
| malignant | 8 | 23 | 1 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8382    0.8636    0.8507        66
   malignant     0.8214    0.7188    0.7667        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8220       118
   macro avg     0.8108    0.8108    0.8090       118
weighted avg     0.8226    0.8220    0.8210       118

```
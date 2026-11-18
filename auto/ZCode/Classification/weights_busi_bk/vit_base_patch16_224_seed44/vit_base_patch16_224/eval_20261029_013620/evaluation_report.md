# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8493 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9448 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8750 |
| Balanced Acc | 0.8314 |
| Macro-Sensitivity | 0.8314 |
| Macro-Specificity | 0.9229 |
| Macro-Precision | 0.8785 |
| AUC-PR (Macro) | 0.8870 |
| Cohen's Kappa | 0.7859 |
| MCC | 0.7927 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 842.9 (1.19 ms/张) |
| End-to-End FPS | 159.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9846 | 0.8235 | 0.8767 | 0.9275 | 0.9767 | 0.9599 | 0.9534 | 65 |
| malignant | 0.7097 | 0.9765 | 0.9167 | 0.8000 | 0.9022 | 0.9317 | 0.9007 | 31 |
| normal | 0.8000 | 0.9688 | 0.8421 | 0.8205 | 0.9588 | 0.9427 | 0.8071 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=64 FP=9 TN=42 FN=1
- **malignant**: TP=22 FP=2 TN=83 FN=9
- **normal**: TP=16 FP=3 TN=93 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 64 | 1 | 0 |
| malignant | 6 | 22 | 3 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8767    0.9846    0.9275        65
   malignant     0.9167    0.7097    0.8000        31
      normal     0.8421    0.8000    0.8205        20

    accuracy                         0.8793       116
   macro avg     0.8785    0.8314    0.8493       116
weighted avg     0.8814    0.8793    0.8750       116

```
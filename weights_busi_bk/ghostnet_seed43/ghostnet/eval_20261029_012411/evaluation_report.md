# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8515 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9657 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8715 |
| Balanced Acc | 0.8603 |
| Macro-Sensitivity | 0.8603 |
| Macro-Specificity | 0.9304 |
| Macro-Precision | 0.8440 |
| AUC-PR (Macro) | 0.9282 |
| Cohen's Kappa | 0.7814 |
| MCC | 0.7818 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1367.4 (0.73 ms/张) |
| End-to-End FPS | 170.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.9020 | 0.9206 | 0.9062 | 0.8679 | 0.9581 | 0.9709 | 65 |
| malignant | 0.8387 | 0.9412 | 0.8387 | 0.8387 | 0.9412 | 0.9583 | 0.8885 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9807 | 0.9251 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=5 TN=46 FN=7
- **malignant**: TP=26 FP=5 TN=80 FN=5
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 5 | 2 |
| malignant | 2 | 26 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9206    0.8923    0.9062        65
   malignant     0.8387    0.8387    0.8387        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8707       116
   macro avg     0.8440    0.8603    0.8515       116
weighted avg     0.8732    0.8707    0.8715       116

```
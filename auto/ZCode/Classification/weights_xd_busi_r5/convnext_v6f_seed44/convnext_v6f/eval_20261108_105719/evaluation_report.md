# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8690 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9501 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8955 |
| Balanced Acc | 0.8645 |
| Macro-Sensitivity | 0.8645 |
| Macro-Specificity | 0.9395 |
| Macro-Precision | 0.8819 |
| AUC-PR (Macro) | 0.9012 |
| Cohen's Kappa | 0.8205 |
| MCC | 0.8236 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 1253.6 (0.80 ms/张) |
| End-to-End FPS | 169.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8824 | 0.9130 | 0.9403 | 0.9574 | 0.9680 | 0.9572 | 65 |
| malignant | 0.7742 | 0.9882 | 0.9600 | 0.8571 | 0.9231 | 0.9480 | 0.9219 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9344 | 0.8245 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=6 TN=45 FN=2
- **malignant**: TP=24 FP=1 TN=84 FN=7
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 1 | 1 |
| malignant | 3 | 24 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9130    0.9692    0.9403        65
   malignant     0.9600    0.7742    0.8571        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8966       116
   macro avg     0.8819    0.8645    0.8690       116
weighted avg     0.9014    0.8966    0.8955       116

```
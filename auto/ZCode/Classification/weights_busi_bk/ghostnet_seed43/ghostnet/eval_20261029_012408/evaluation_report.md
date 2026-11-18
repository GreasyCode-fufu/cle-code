# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7562 |
| Accuracy | 0.7712 |
| Macro-AUROC (OvR) | 0.9219 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7734 |
| Balanced Acc | 0.7796 |
| Macro-Sensitivity | 0.7796 |
| Macro-Specificity | 0.8768 |
| Macro-Precision | 0.7409 |
| AUC-PR (Macro) | 0.8449 |
| Cohen's Kappa | 0.6226 |
| MCC | 0.6258 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1388.1 (0.72 ms/张) |
| End-to-End FPS | 173.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7576 | 0.8269 | 0.8475 | 0.8000 | 0.7288 | 0.8820 | 0.9247 | 66 |
| malignant | 0.7812 | 0.8953 | 0.7353 | 0.7576 | 0.9167 | 0.9219 | 0.7521 | 32 |
| normal | 0.8000 | 0.9082 | 0.6400 | 0.7111 | 0.9570 | 0.9617 | 0.8578 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=50 FP=9 TN=43 FN=16
- **malignant**: TP=25 FP=9 TN=77 FN=7
- **normal**: TP=16 FP=9 TN=89 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 50 | 9 | 7 |
| malignant | 5 | 25 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8475    0.7576    0.8000        66
   malignant     0.7353    0.7812    0.7576        32
      normal     0.6400    0.8000    0.7111        20

    accuracy                         0.7712       118
   macro avg     0.7409    0.7796    0.7562       118
weighted avg     0.7819    0.7712    0.7734       118

```
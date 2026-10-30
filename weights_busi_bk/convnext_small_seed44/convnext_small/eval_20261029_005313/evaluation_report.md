# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8578 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9531 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8890 |
| Balanced Acc | 0.8709 |
| Macro-Sensitivity | 0.8709 |
| Macro-Specificity | 0.9443 |
| Macro-Precision | 0.8540 |
| AUC-PR (Macro) | 0.8925 |
| Cohen's Kappa | 0.8105 |
| MCC | 0.8127 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 907.7 (1.10 ms/张) |
| End-to-End FPS | 161.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9412 | 0.9531 | 0.9457 | 0.9231 | 0.9692 | 0.9689 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9431 | 0.9021 | 31 |
| normal | 0.9000 | 0.9271 | 0.7200 | 0.8000 | 0.9780 | 0.9469 | 0.8064 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=3 TN=48 FN=4
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=18 FP=7 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 1 | 24 | 6 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9531    0.9385    0.9457        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8879       116
   macro avg     0.8540    0.8709    0.8578       116
weighted avg     0.8958    0.8879    0.8890       116

```
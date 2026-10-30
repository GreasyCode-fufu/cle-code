# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7950 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9424 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8080 |
| Balanced Acc | 0.8400 |
| Macro-Sensitivity | 0.8400 |
| Macro-Specificity | 0.9055 |
| Macro-Precision | 0.7747 |
| AUC-PR (Macro) | 0.8801 |
| Cohen's Kappa | 0.6866 |
| MCC | 0.6968 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1467.6 (0.68 ms/张) |
| End-to-End FPS | 180.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7576 | 0.9231 | 0.9259 | 0.8333 | 0.7500 | 0.9164 | 0.9485 | 66 |
| malignant | 0.8125 | 0.8953 | 0.7429 | 0.7761 | 0.9277 | 0.9306 | 0.7847 | 32 |
| normal | 0.9500 | 0.8980 | 0.6552 | 0.7755 | 0.9888 | 0.9801 | 0.9071 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=50 FP=4 TN=48 FN=16
- **malignant**: TP=26 FP=9 TN=77 FN=6
- **normal**: TP=19 FP=10 TN=88 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 50 | 9 | 7 |
| malignant | 3 | 26 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9259    0.7576    0.8333        66
   malignant     0.7429    0.8125    0.7761        32
      normal     0.6552    0.9500    0.7755        20

    accuracy                         0.8051       118
   macro avg     0.7747    0.8400    0.7950       118
weighted avg     0.8304    0.8051    0.8080       118

```
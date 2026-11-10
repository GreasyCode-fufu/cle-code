# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/jumbo_small_seed44/jumbo_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8041 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9377 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8239 |
| Balanced Acc | 0.8278 |
| Macro-Sensitivity | 0.8278 |
| Macro-Specificity | 0.9051 |
| Macro-Precision | 0.7938 |
| AUC-PR (Macro) | 0.8637 |
| Cohen's Kappa | 0.7037 |
| MCC | 0.7070 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1427.4 (0.70 ms/张) |
| End-to-End FPS | 174.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.8654 | 0.8871 | 0.8594 | 0.8036 | 0.9286 | 0.9514 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9132 | 0.8350 | 32 |
| normal | 0.9000 | 0.9082 | 0.6667 | 0.7660 | 0.9780 | 0.9714 | 0.8047 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=7 TN=45 FN=11
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=18 FP=9 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 5 | 6 |
| malignant | 5 | 24 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8871    0.8333    0.8594        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.6667    0.9000    0.7660        20

    accuracy                         0.8220       118
   macro avg     0.7938    0.8278    0.8041       118
weighted avg     0.8336    0.8220    0.8239       118

```
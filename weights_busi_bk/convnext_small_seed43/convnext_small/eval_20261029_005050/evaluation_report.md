# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8213 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9456 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8403 |
| Balanced Acc | 0.8361 |
| Macro-Sensitivity | 0.8361 |
| Macro-Specificity | 0.9140 |
| Macro-Precision | 0.8096 |
| AUC-PR (Macro) | 0.9060 |
| Cohen's Kappa | 0.7311 |
| MCC | 0.7329 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 908.8 (1.10 ms/张) |
| End-to-End FPS | 168.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.8846 | 0.9016 | 0.8661 | 0.8070 | 0.9238 | 0.9406 | 66 |
| malignant | 0.8750 | 0.9186 | 0.8000 | 0.8358 | 0.9518 | 0.9528 | 0.8805 | 32 |
| normal | 0.8000 | 0.9388 | 0.7273 | 0.7619 | 0.9583 | 0.9602 | 0.8969 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=6 TN=46 FN=11
- **malignant**: TP=28 FP=7 TN=79 FN=4
- **normal**: TP=16 FP=6 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 6 | 5 |
| malignant | 3 | 28 | 1 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9016    0.8333    0.8661        66
   malignant     0.8000    0.8750    0.8358        32
      normal     0.7273    0.8000    0.7619        20

    accuracy                         0.8390       118
   macro avg     0.8096    0.8361    0.8213       118
weighted avg     0.8445    0.8390    0.8403       118

```
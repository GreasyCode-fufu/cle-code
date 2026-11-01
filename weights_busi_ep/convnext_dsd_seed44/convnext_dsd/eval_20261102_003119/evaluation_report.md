# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dsd_seed44/convnext_dsd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7983 |
| Accuracy | 0.8136 |
| Macro-AUROC (OvR) | 0.9362 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8149 |
| Balanced Acc | 0.8227 |
| Macro-Sensitivity | 0.8227 |
| Macro-Specificity | 0.9008 |
| Macro-Precision | 0.7839 |
| AUC-PR (Macro) | 0.8815 |
| Cohen's Kappa | 0.6904 |
| MCC | 0.6932 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 420.2 (2.38 ms/张) |
| End-to-End FPS | 135.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.8654 | 0.8852 | 0.8504 | 0.7895 | 0.9337 | 0.9518 | 66 |
| malignant | 0.7500 | 0.9186 | 0.7742 | 0.7619 | 0.9080 | 0.9244 | 0.8576 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9505 | 0.8352 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=7 TN=45 FN=12
- **malignant**: TP=24 FP=7 TN=79 FN=8
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 7 | 5 |
| malignant | 5 | 24 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8852    0.8182    0.8504        66
   malignant     0.7742    0.7500    0.7619        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8136       118
   macro avg     0.7839    0.8227    0.7983       118
weighted avg     0.8224    0.8136    0.8149       118

```
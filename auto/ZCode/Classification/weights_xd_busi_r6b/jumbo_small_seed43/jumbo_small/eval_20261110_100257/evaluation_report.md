# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/jumbo_small_seed43/jumbo_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8851 |
| Accuracy | 0.8983 |
| Macro-AUROC (OvR) | 0.9529 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8998 |
| Balanced Acc | 0.9063 |
| Macro-Sensitivity | 0.9063 |
| Macro-Specificity | 0.9487 |
| Macro-Precision | 0.8716 |
| AUC-PR (Macro) | 0.8926 |
| Cohen's Kappa | 0.8302 |
| MCC | 0.8325 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1409.0 (0.71 ms/张) |
| End-to-End FPS | 172.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.9423 | 0.9516 | 0.9219 | 0.8750 | 0.9452 | 0.9626 | 66 |
| malignant | 0.8750 | 0.9651 | 0.9032 | 0.8889 | 0.9540 | 0.9506 | 0.8863 | 32 |
| normal | 0.9500 | 0.9388 | 0.7600 | 0.8444 | 0.9892 | 0.9628 | 0.8290 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=3 TN=49 FN=7
- **malignant**: TP=28 FP=3 TN=83 FN=4
- **normal**: TP=19 FP=6 TN=92 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 3 | 4 |
| malignant | 2 | 28 | 2 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9516    0.8939    0.9219        66
   malignant     0.9032    0.8750    0.8889        32
      normal     0.7600    0.9500    0.8444        20

    accuracy                         0.8983       118
   macro avg     0.8716    0.9063    0.8851       118
weighted avg     0.9060    0.8983    0.8998       118

```
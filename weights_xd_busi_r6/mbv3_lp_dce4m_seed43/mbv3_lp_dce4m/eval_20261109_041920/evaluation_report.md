# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_lp_dce4m_seed43/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7726 |
| Accuracy | 0.7881 |
| Macro-AUROC (OvR) | 0.9017 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7864 |
| Balanced Acc | 0.7799 |
| Macro-Sensitivity | 0.7799 |
| Macro-Specificity | 0.8755 |
| Macro-Precision | 0.7700 |
| AUC-PR (Macro) | 0.8523 |
| Cohen's Kappa | 0.6379 |
| MCC | 0.6389 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1044.2 (0.96 ms/张) |
| End-to-End FPS | 15.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.7692 | 0.8209 | 0.8271 | 0.7843 | 0.8706 | 0.8976 | 66 |
| malignant | 0.6562 | 0.9186 | 0.7500 | 0.7000 | 0.8778 | 0.8721 | 0.7478 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9622 | 0.9115 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=12 TN=40 FN=11
- **malignant**: TP=21 FP=7 TN=79 FN=11
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 7 | 4 |
| malignant | 9 | 21 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8209    0.8333    0.8271        66
   malignant     0.7500    0.6562    0.7000        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.7881       118
   macro avg     0.7700    0.7799    0.7726       118
weighted avg     0.7878    0.7881    0.7864       118

```
# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8420 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9538 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8643 |
| Balanced Acc | 0.8396 |
| Macro-Sensitivity | 0.8396 |
| Macro-Specificity | 0.9242 |
| Macro-Precision | 0.8470 |
| AUC-PR (Macro) | 0.9092 |
| Cohen's Kappa | 0.7685 |
| MCC | 0.7691 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1182.1 (0.85 ms/张) |
| End-to-End FPS | 7.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8846 | 0.9077 | 0.9008 | 0.8679 | 0.9493 | 0.9661 | 66 |
| malignant | 0.8750 | 0.9186 | 0.8000 | 0.8358 | 0.9518 | 0.9520 | 0.8848 | 32 |
| normal | 0.7500 | 0.9694 | 0.8333 | 0.7895 | 0.9500 | 0.9602 | 0.8767 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=6 TN=46 FN=7
- **malignant**: TP=28 FP=7 TN=79 FN=4
- **normal**: TP=15 FP=3 TN=95 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 7 | 0 |
| malignant | 1 | 28 | 3 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9077    0.8939    0.9008        66
   malignant     0.8000    0.8750    0.8358        32
      normal     0.8333    0.7500    0.7895        20

    accuracy                         0.8644       118
   macro avg     0.8470    0.8396    0.8420       118
weighted avg     0.8659    0.8644    0.8643       118

```
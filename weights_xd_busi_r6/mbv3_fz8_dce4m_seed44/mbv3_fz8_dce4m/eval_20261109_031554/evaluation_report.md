# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8777 |
| Accuracy | 0.8814 |
| Macro-AUROC (OvR) | 0.9637 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8806 |
| Balanced Acc | 0.8685 |
| Macro-Sensitivity | 0.8685 |
| Macro-Specificity | 0.9259 |
| Macro-Precision | 0.8883 |
| AUC-PR (Macro) | 0.9339 |
| Cohen's Kappa | 0.7943 |
| MCC | 0.7951 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1538.4 (0.65 ms/张) |
| End-to-End FPS | 12.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9242 | 0.8462 | 0.8841 | 0.9037 | 0.8980 | 0.9525 | 0.9700 | 66 |
| malignant | 0.7812 | 0.9419 | 0.8333 | 0.8065 | 0.9205 | 0.9564 | 0.8805 | 32 |
| normal | 0.9000 | 0.9898 | 0.9474 | 0.9231 | 0.9798 | 0.9821 | 0.9512 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=8 TN=44 FN=5
- **malignant**: TP=25 FP=5 TN=81 FN=7
- **normal**: TP=18 FP=1 TN=97 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 5 | 0 |
| malignant | 6 | 25 | 1 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8841    0.9242    0.9037        66
   malignant     0.8333    0.7812    0.8065        32
      normal     0.9474    0.9000    0.9231        20

    accuracy                         0.8814       118
   macro avg     0.8883    0.8685    0.8777       118
weighted avg     0.8810    0.8814    0.8806       118

```
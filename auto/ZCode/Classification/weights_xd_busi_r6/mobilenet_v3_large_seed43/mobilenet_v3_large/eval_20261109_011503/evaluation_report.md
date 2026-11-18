# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7910 |
| Accuracy | 0.8136 |
| Macro-AUROC (OvR) | 0.9435 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8165 |
| Balanced Acc | 0.8111 |
| Macro-Sensitivity | 0.8111 |
| Macro-Specificity | 0.9054 |
| Macro-Precision | 0.7770 |
| AUC-PR (Macro) | 0.8871 |
| Cohen's Kappa | 0.6908 |
| MCC | 0.6934 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1546.7 (0.65 ms/张) |
| End-to-End FPS | 172.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.9038 | 0.9167 | 0.8730 | 0.8103 | 0.9473 | 0.9626 | 66 |
| malignant | 0.7500 | 0.8837 | 0.7059 | 0.7273 | 0.9048 | 0.9124 | 0.8302 | 32 |
| normal | 0.8500 | 0.9286 | 0.7083 | 0.7727 | 0.9681 | 0.9709 | 0.8683 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=5 TN=47 FN=11
- **malignant**: TP=24 FP=10 TN=76 FN=8
- **normal**: TP=17 FP=7 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 8 | 3 |
| malignant | 4 | 24 | 4 |
| normal | 1 | 2 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9167    0.8333    0.8730        66
   malignant     0.7059    0.7500    0.7273        32
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.8136       118
   macro avg     0.7770    0.8111    0.7910       118
weighted avg     0.8242    0.8136    0.8165       118

```
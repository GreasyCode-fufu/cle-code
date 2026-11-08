# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8366 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9478 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8559 |
| Balanced Acc | 0.8417 |
| Macro-Sensitivity | 0.8417 |
| Macro-Specificity | 0.9187 |
| Macro-Precision | 0.8333 |
| AUC-PR (Macro) | 0.8996 |
| Cohen's Kappa | 0.7544 |
| MCC | 0.7548 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1695.5 (0.59 ms/张) |
| End-to-End FPS | 159.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8654 | 0.8939 | 0.8939 | 0.8654 | 0.9378 | 0.9547 | 66 |
| malignant | 0.7812 | 0.9419 | 0.8333 | 0.8065 | 0.9205 | 0.9509 | 0.8729 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9546 | 0.8711 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=45 FN=7
- **malignant**: TP=25 FP=5 TN=81 FN=7
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 5 | 2 |
| malignant | 4 | 25 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.8939    0.8939        66
   malignant     0.8333    0.7812    0.8065        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8559       118
   macro avg     0.8333    0.8417    0.8366       118
weighted avg     0.8570    0.8559    0.8559       118

```
# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8066 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9456 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8324 |
| Balanced Acc | 0.8212 |
| Macro-Sensitivity | 0.8212 |
| Macro-Specificity | 0.9127 |
| Macro-Precision | 0.7953 |
| AUC-PR (Macro) | 0.8883 |
| Cohen's Kappa | 0.7162 |
| MCC | 0.7175 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1710.1 (0.58 ms/张) |
| End-to-End FPS | 154.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.9038 | 0.9194 | 0.8906 | 0.8393 | 0.9435 | 0.9628 | 66 |
| malignant | 0.7500 | 0.8953 | 0.7273 | 0.7385 | 0.9059 | 0.9215 | 0.8333 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9719 | 0.8686 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=47 FN=9
- **malignant**: TP=24 FP=9 TN=77 FN=8
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 7 | 2 |
| malignant | 4 | 24 | 4 |
| normal | 1 | 2 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8636    0.8906        66
   malignant     0.7273    0.7500    0.7385        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8305       118
   macro avg     0.7953    0.8212    0.8066       118
weighted avg     0.8367    0.8305    0.8324       118

```
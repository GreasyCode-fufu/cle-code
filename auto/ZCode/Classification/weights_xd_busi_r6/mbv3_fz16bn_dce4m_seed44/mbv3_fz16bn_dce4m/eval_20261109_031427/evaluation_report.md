# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed44/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8333 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9533 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8561 |
| Balanced Acc | 0.8355 |
| Macro-Sensitivity | 0.8355 |
| Macro-Specificity | 0.9187 |
| Macro-Precision | 0.8315 |
| AUC-PR (Macro) | 0.9069 |
| Cohen's Kappa | 0.7540 |
| MCC | 0.7541 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1584.9 (0.63 ms/张) |
| End-to-End FPS | 12.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8654 | 0.8939 | 0.8939 | 0.8654 | 0.9441 | 0.9632 | 66 |
| malignant | 0.8125 | 0.9419 | 0.8387 | 0.8254 | 0.9310 | 0.9526 | 0.8777 | 32 |
| normal | 0.8000 | 0.9490 | 0.7619 | 0.7805 | 0.9588 | 0.9633 | 0.8798 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=45 FN=7
- **malignant**: TP=26 FP=5 TN=81 FN=6
- **normal**: TP=16 FP=5 TN=93 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 5 | 2 |
| malignant | 3 | 26 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.8939    0.8939        66
   malignant     0.8387    0.8125    0.8254        32
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8559       118
   macro avg     0.8315    0.8355    0.8333       118
weighted avg     0.8566    0.8559    0.8561       118

```
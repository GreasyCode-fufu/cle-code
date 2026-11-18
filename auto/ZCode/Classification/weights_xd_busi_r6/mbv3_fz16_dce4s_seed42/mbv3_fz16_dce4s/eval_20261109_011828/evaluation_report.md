# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7962 |
| Accuracy | 0.8362 |
| Macro-AUROC (OvR) | 0.9274 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8350 |
| Balanced Acc | 0.7943 |
| Macro-Sensitivity | 0.7943 |
| Macro-Specificity | 0.9077 |
| Macro-Precision | 0.8034 |
| AUC-PR (Macro) | 0.8755 |
| Cohen's Kappa | 0.7171 |
| MCC | 0.7189 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1706.9 (0.59 ms/张) |
| End-to-End FPS | 150.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.8431 | 0.8824 | 0.9023 | 0.8958 | 0.9376 | 0.9369 | 65 |
| malignant | 0.7097 | 0.9529 | 0.8462 | 0.7719 | 0.9000 | 0.9180 | 0.8578 | 31 |
| normal | 0.7500 | 0.9271 | 0.6818 | 0.7143 | 0.9468 | 0.9266 | 0.8319 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=8 TN=43 FN=5
- **malignant**: TP=22 FP=4 TN=81 FN=9
- **normal**: TP=15 FP=7 TN=89 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 3 | 2 |
| malignant | 4 | 22 | 5 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8824    0.9231    0.9023        65
   malignant     0.8462    0.7097    0.7719        31
      normal     0.6818    0.7500    0.7143        20

    accuracy                         0.8362       116
   macro avg     0.8034    0.7943    0.7962       116
weighted avg     0.8381    0.8362    0.8350       116

```
# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8860 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9670 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8764 |
| Balanced Acc | 0.8835 |
| Macro-Sensitivity | 0.8835 |
| Macro-Specificity | 0.9306 |
| Macro-Precision | 0.8895 |
| AUC-PR (Macro) | 0.9497 |
| Cohen's Kappa | 0.8043 |
| MCC | 0.8050 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3392.3 (0.29 ms/张) |
| End-to-End FPS | 19.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8020 | 0.9369 | 0.8634 | 0.8316 | 0.9049 | 0.9534 | 0.9220 | 197 |
| normal | 0.8986 | 0.8675 | 0.8552 | 0.8763 | 0.9076 | 0.9512 | 0.9436 | 276 |
| sever | 0.9500 | 0.9873 | 0.9500 | 0.9500 | 0.9873 | 0.9964 | 0.9834 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=158 FP=25 TN=371 FN=39
- **normal**: TP=248 FP=42 TN=275 FN=28
- **sever**: TP=114 FP=6 TN=467 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 158 | 38 | 1 |
| normal | 23 | 248 | 5 |
| sever | 2 | 4 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8634    0.8020    0.8316       197
      normal     0.8552    0.8986    0.8763       276
       sever     0.9500    0.9500    0.9500       120

    accuracy                         0.8769       593
   macro avg     0.8895    0.8835    0.8860       593
weighted avg     0.8771    0.8769    0.8764       593

```
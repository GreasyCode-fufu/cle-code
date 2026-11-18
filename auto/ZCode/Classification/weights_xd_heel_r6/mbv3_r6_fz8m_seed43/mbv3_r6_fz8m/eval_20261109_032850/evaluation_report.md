# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7807 |
| Accuracy | 0.7723 |
| Macro-AUROC (OvR) | 0.9350 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7687 |
| Balanced Acc | 0.8163 |
| Macro-Sensitivity | 0.8163 |
| Macro-Specificity | 0.8874 |
| Macro-Precision | 0.7734 |
| AUC-PR (Macro) | 0.9002 |
| Cohen's Kappa | 0.6540 |
| MCC | 0.6666 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4110.0 (0.24 ms/张) |
| End-to-End FPS | 150.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.8258 | 0.7039 | 0.7628 | 0.9083 | 0.9135 | 0.8562 | 197 |
| normal | 0.6413 | 0.9274 | 0.8850 | 0.7437 | 0.7481 | 0.9034 | 0.8839 | 276 |
| sever | 0.9750 | 0.9091 | 0.7312 | 0.8357 | 0.9931 | 0.9880 | 0.9606 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=69 TN=327 FN=33
- **normal**: TP=177 FP=23 TN=294 FN=99
- **sever**: TP=117 FP=43 TN=430 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 20 | 13 |
| normal | 69 | 177 | 30 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7039    0.8325    0.7628       197
      normal     0.8850    0.6413    0.7437       276
       sever     0.7312    0.9750    0.8357       120

    accuracy                         0.7723       593
   macro avg     0.7734    0.8163    0.7807       593
weighted avg     0.7937    0.7723    0.7687       593

```
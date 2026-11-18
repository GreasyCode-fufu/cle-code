# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7735 |
| Accuracy | 0.7555 |
| Macro-AUROC (OvR) | 0.9171 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7514 |
| Balanced Acc | 0.8033 |
| Macro-Sensitivity | 0.8033 |
| Macro-Specificity | 0.8754 |
| Macro-Precision | 0.7670 |
| AUC-PR (Macro) | 0.8723 |
| Cohen's Kappa | 0.6265 |
| MCC | 0.6382 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3302.7 (0.30 ms/张) |
| End-to-End FPS | 274.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8071 | 0.7854 | 0.6516 | 0.7211 | 0.8911 | 0.8801 | 0.8026 | 197 |
| normal | 0.6196 | 0.9022 | 0.8465 | 0.7155 | 0.7315 | 0.8816 | 0.8521 | 276 |
| sever | 0.9833 | 0.9387 | 0.8027 | 0.8839 | 0.9955 | 0.9897 | 0.9621 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=159 FP=85 TN=311 FN=38
- **normal**: TP=171 FP=31 TN=286 FN=105
- **sever**: TP=118 FP=29 TN=444 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 159 | 29 | 9 |
| normal | 85 | 171 | 20 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6516    0.8071    0.7211       197
      normal     0.8465    0.6196    0.7155       276
       sever     0.8027    0.9833    0.8839       120

    accuracy                         0.7555       593
   macro avg     0.7670    0.8033    0.7735       593
weighted avg     0.7729    0.7555    0.7514       593

```
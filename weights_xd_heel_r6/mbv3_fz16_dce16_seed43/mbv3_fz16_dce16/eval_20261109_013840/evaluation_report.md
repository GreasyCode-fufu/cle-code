# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8452 |
| Accuracy | 0.8314 |
| Macro-AUROC (OvR) | 0.9392 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8305 |
| Balanced Acc | 0.8472 |
| Macro-Sensitivity | 0.8472 |
| Macro-Specificity | 0.9066 |
| Macro-Precision | 0.8438 |
| AUC-PR (Macro) | 0.9041 |
| Cohen's Kappa | 0.7331 |
| MCC | 0.7333 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3675.0 (0.27 ms/张) |
| End-to-End FPS | 11.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7462 | 0.8965 | 0.7819 | 0.7636 | 0.8765 | 0.9112 | 0.8378 | 197 |
| normal | 0.8370 | 0.8423 | 0.8221 | 0.8294 | 0.8558 | 0.9105 | 0.8914 | 276 |
| sever | 0.9583 | 0.9810 | 0.9274 | 0.9426 | 0.9893 | 0.9960 | 0.9831 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=147 FP=41 TN=355 FN=50
- **normal**: TP=231 FP=50 TN=267 FN=45
- **sever**: TP=115 FP=9 TN=464 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 147 | 47 | 3 |
| normal | 39 | 231 | 6 |
| sever | 2 | 3 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7819    0.7462    0.7636       197
      normal     0.8221    0.8370    0.8294       276
       sever     0.9274    0.9583    0.9426       120

    accuracy                         0.8314       593
   macro avg     0.8438    0.8472    0.8452       593
weighted avg     0.8300    0.8314    0.8305       593

```
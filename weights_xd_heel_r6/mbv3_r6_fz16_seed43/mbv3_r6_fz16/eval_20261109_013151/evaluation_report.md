# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7268 |
| Accuracy | 0.7099 |
| Macro-AUROC (OvR) | 0.9226 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7000 |
| Balanced Acc | 0.7716 |
| Macro-Sensitivity | 0.7716 |
| Macro-Specificity | 0.8563 |
| Macro-Precision | 0.7371 |
| AUC-PR (Macro) | 0.8831 |
| Cohen's Kappa | 0.5645 |
| MCC | 0.5906 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3192.3 (0.31 ms/张) |
| End-to-End FPS | 105.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.7197 | 0.5964 | 0.6949 | 0.8962 | 0.8789 | 0.8069 | 197 |
| normal | 0.5072 | 0.9338 | 0.8696 | 0.6407 | 0.6852 | 0.8954 | 0.8631 | 276 |
| sever | 0.9750 | 0.9154 | 0.7452 | 0.8448 | 0.9931 | 0.9935 | 0.9794 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=111 TN=285 FN=33
- **normal**: TP=140 FP=21 TN=296 FN=136
- **sever**: TP=117 FP=40 TN=433 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 20 | 13 |
| normal | 109 | 140 | 27 |
| sever | 2 | 1 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.5964    0.8325    0.6949       197
      normal     0.8696    0.5072    0.6407       276
       sever     0.7452    0.9750    0.8448       120

    accuracy                         0.7099       593
   macro avg     0.7371    0.7716    0.7268       593
weighted avg     0.7536    0.7099    0.7000       593

```
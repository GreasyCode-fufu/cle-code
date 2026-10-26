# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8935 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8779 |
| Balanced Acc | 0.8979 |
| Macro-Sensitivity | 0.8979 |
| Macro-Specificity | 0.9345 |
| Macro-Precision | 0.8946 |
| AUC-PR (Macro) | 0.9686 |
| Cohen's Kappa | 0.8072 |
| MCC | 0.8109 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 576.6 (1.73 ms/张) |
| End-to-End FPS | 237.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9036 | 0.8687 | 0.7739 | 0.8337 | 0.9477 | 0.9666 | 0.9442 | 197 |
| normal | 0.8152 | 0.9369 | 0.9184 | 0.8637 | 0.8534 | 0.9645 | 0.9620 | 276 |
| sever | 0.9750 | 0.9979 | 0.9915 | 0.9832 | 0.9937 | 0.9999 | 0.9995 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=178 FP=52 TN=344 FN=19
- **normal**: TP=225 FP=20 TN=297 FN=51
- **sever**: TP=117 FP=1 TN=472 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 178 | 18 | 1 |
| normal | 51 | 225 | 0 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7739    0.9036    0.8337       197
      normal     0.9184    0.8152    0.8637       276
       sever     0.9915    0.9750    0.9832       120

    accuracy                         0.8769       593
   macro avg     0.8946    0.8979    0.8935       593
weighted avg     0.8852    0.8769    0.8779       593

```
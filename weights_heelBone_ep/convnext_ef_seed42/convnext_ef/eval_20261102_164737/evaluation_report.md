# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8862 |
| Accuracy | 0.8820 |
| Macro-AUROC (OvR) | 0.9739 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8813 |
| Balanced Acc | 0.8994 |
| Macro-Sensitivity | 0.8994 |
| Macro-Specificity | 0.9386 |
| Macro-Precision | 0.8768 |
| AUC-PR (Macro) | 0.9635 |
| Cohen's Kappa | 0.8154 |
| MCC | 0.8165 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 593.8 (1.68 ms/张) |
| End-to-End FPS | 242.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8477 | 0.9369 | 0.8698 | 0.8586 | 0.9252 | 0.9573 | 0.9404 | 197 |
| normal | 0.8587 | 0.9211 | 0.9046 | 0.8810 | 0.8822 | 0.9659 | 0.9555 | 276 |
| sever | 0.9917 | 0.9577 | 0.8561 | 0.9189 | 0.9978 | 0.9985 | 0.9945 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=167 FP=25 TN=371 FN=30
- **normal**: TP=237 FP=25 TN=292 FN=39
- **sever**: TP=119 FP=20 TN=453 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 167 | 24 | 6 |
| normal | 25 | 237 | 14 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8698    0.8477    0.8586       197
      normal     0.9046    0.8587    0.8810       276
       sever     0.8561    0.9917    0.9189       120

    accuracy                         0.8820       593
   macro avg     0.8768    0.8994    0.8862       593
weighted avg     0.8832    0.8820    0.8813       593

```
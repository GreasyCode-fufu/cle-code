# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8694 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9742 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8577 |
| Balanced Acc | 0.8844 |
| Macro-Sensitivity | 0.8844 |
| Macro-Specificity | 0.9267 |
| Macro-Precision | 0.8604 |
| AUC-PR (Macro) | 0.9615 |
| Cohen's Kappa | 0.7797 |
| MCC | 0.7830 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 597.3 (1.67 ms/张) |
| End-to-End FPS | 239.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.8813 | 0.7844 | 0.8241 | 0.9307 | 0.9622 | 0.9342 | 197 |
| normal | 0.7935 | 0.9306 | 0.9087 | 0.8472 | 0.8381 | 0.9618 | 0.9566 | 276 |
| sever | 0.9917 | 0.9683 | 0.8881 | 0.9370 | 0.9978 | 0.9985 | 0.9937 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=47 TN=349 FN=26
- **normal**: TP=219 FP=22 TN=295 FN=57
- **sever**: TP=119 FP=15 TN=458 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 21 | 5 |
| normal | 47 | 219 | 10 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7844    0.8680    0.8241       197
      normal     0.9087    0.7935    0.8472       276
       sever     0.8881    0.9917    0.9370       120

    accuracy                         0.8583       593
   macro avg     0.8604    0.8844    0.8694       593
weighted avg     0.8632    0.8583    0.8577       593

```
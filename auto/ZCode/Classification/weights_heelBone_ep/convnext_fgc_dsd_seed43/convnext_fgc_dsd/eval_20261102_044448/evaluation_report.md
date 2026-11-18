# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8784 |
| Accuracy | 0.8668 |
| Macro-AUROC (OvR) | 0.9780 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8666 |
| Balanced Acc | 0.8912 |
| Macro-Sensitivity | 0.8912 |
| Macro-Specificity | 0.9314 |
| Macro-Precision | 0.8733 |
| AUC-PR (Macro) | 0.9676 |
| Cohen's Kappa | 0.7928 |
| MCC | 0.7979 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 879.8 (1.14 ms/张) |
| End-to-End FPS | 278.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9086 | 0.8712 | 0.7783 | 0.8384 | 0.9504 | 0.9677 | 0.9506 | 197 |
| normal | 0.7899 | 0.9464 | 0.9277 | 0.8532 | 0.8380 | 0.9697 | 0.9628 | 276 |
| sever | 0.9750 | 0.9767 | 0.9141 | 0.9435 | 0.9935 | 0.9967 | 0.9893 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=179 FP=51 TN=345 FN=18
- **normal**: TP=218 FP=17 TN=300 FN=58
- **sever**: TP=117 FP=11 TN=462 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 179 | 15 | 3 |
| normal | 50 | 218 | 8 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7783    0.9086    0.8384       197
      normal     0.9277    0.7899    0.8532       276
       sever     0.9141    0.9750    0.9435       120

    accuracy                         0.8668       593
   macro avg     0.8733    0.8912    0.8784       593
weighted avg     0.8753    0.8668    0.8666       593

```
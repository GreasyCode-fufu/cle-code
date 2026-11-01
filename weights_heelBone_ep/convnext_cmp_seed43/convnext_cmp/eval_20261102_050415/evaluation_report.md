# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed43/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9038 |
| Accuracy | 0.8921 |
| Macro-AUROC (OvR) | 0.9827 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8922 |
| Balanced Acc | 0.9095 |
| Macro-Sensitivity | 0.9095 |
| Macro-Specificity | 0.9422 |
| Macro-Precision | 0.8995 |
| AUC-PR (Macro) | 0.9752 |
| Cohen's Kappa | 0.8305 |
| MCC | 0.8314 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 1685.3 (0.59 ms/张) |
| End-to-End FPS | 332.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8782 | 0.9066 | 0.8238 | 0.8501 | 0.9373 | 0.9742 | 0.9562 | 197 |
| normal | 0.8587 | 0.9306 | 0.9151 | 0.8860 | 0.8832 | 0.9743 | 0.9708 | 276 |
| sever | 0.9917 | 0.9894 | 0.9597 | 0.9754 | 0.9979 | 0.9997 | 0.9988 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=173 FP=37 TN=359 FN=24
- **normal**: TP=237 FP=22 TN=295 FN=39
- **sever**: TP=119 FP=5 TN=468 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 173 | 22 | 2 |
| normal | 36 | 237 | 3 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8238    0.8782    0.8501       197
      normal     0.9151    0.8587    0.8860       276
       sever     0.9597    0.9917    0.9754       120

    accuracy                         0.8921       593
   macro avg     0.8995    0.9095    0.9038       593
weighted avg     0.8938    0.8921    0.8922       593

```
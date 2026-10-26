# resnext50_32x4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnext50_32x4d_seed42/resnext50_32x4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9296 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9877 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9302 |
| Balanced Acc | 0.9279 |
| Macro-Sensitivity | 0.9279 |
| Macro-Specificity | 0.9647 |
| Macro-Precision | 0.9332 |
| AUC-PR (Macro) | 0.9752 |
| Cohen's Kappa | 0.8954 |
| MCC | 0.8963 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.29G |
| FLOPs | 8.57G |
| Params | 22.99M |
| Pure-Inference FPS | 733.9 (1.36 ms/张) |
| End-to-End FPS | 208.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9348 | 0.8919 | 0.9209 | 0.9718 | 0.9823 | 0.9711 | 208 |
| Malignant_lesions | 0.8580 | 0.9775 | 0.9437 | 0.8988 | 0.9399 | 0.9823 | 0.9572 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9985 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=24 TN=344 FN=10
- **Malignant_lesions**: TP=151 FP=9 TN=391 FN=25
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 21 | 151 | 4 |
| body_of_stomach_normal | 3 | 2 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8919    0.9519    0.9209       208
     Malignant_lesions     0.9437    0.8580    0.8988       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9306       576
             macro avg     0.9332    0.9279    0.9296       576
          weighted avg     0.9317    0.9306    0.9302       576

```
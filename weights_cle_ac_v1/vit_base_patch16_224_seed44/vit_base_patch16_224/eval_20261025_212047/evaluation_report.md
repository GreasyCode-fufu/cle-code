# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9203 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9885 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9201 |
| Balanced Acc | 0.9206 |
| Macro-Sensitivity | 0.9206 |
| Macro-Specificity | 0.9599 |
| Macro-Precision | 0.9242 |
| AUC-PR (Macro) | 0.9792 |
| Cohen's Kappa | 0.8800 |
| MCC | 0.8818 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 263.8 (3.79 ms/张) |
| End-to-End FPS | 129.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9375 | 0.8950 | 0.9180 | 0.9664 | 0.9829 | 0.9742 | 208 |
| Malignant_lesions | 0.9602 | 0.9475 | 0.8895 | 0.9235 | 0.9819 | 0.9885 | 0.9743 | 176 |
| body_of_stomach_normal | 0.8594 | 0.9948 | 0.9880 | 0.9192 | 0.9340 | 0.9940 | 0.9893 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=23 TN=345 FN=12
- **Malignant_lesions**: TP=169 FP=21 TN=379 FN=7
- **body_of_stomach_normal**: TP=165 FP=2 TN=382 FN=27

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 6 | 169 | 1 |
| body_of_stomach_normal | 17 | 10 | 165 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8950    0.9423    0.9180       208
     Malignant_lesions     0.8895    0.9602    0.9235       176
body_of_stomach_normal     0.9880    0.8594    0.9192       192

              accuracy                         0.9201       576
             macro avg     0.9242    0.9206    0.9203       576
          weighted avg     0.9243    0.9201    0.9201       576

```
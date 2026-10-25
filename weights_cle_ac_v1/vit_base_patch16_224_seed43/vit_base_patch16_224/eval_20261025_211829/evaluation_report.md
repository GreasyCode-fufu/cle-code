# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9311 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9919 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9319 |
| Balanced Acc | 0.9304 |
| Macro-Sensitivity | 0.9304 |
| Macro-Specificity | 0.9659 |
| Macro-Precision | 0.9335 |
| AUC-PR (Macro) | 0.9846 |
| Cohen's Kappa | 0.8981 |
| MCC | 0.8988 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 222.0 (4.51 ms/张) |
| End-to-End FPS | 124.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9903 | 0.9848 | 208 |
| Malignant_lesions | 0.8750 | 0.9800 | 0.9506 | 0.9112 | 0.9469 | 0.9897 | 0.9766 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9531 | 0.9122 | 0.9421 | 0.9865 | 0.9956 | 0.9924 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=154 FP=8 TN=392 FN=22
- **body_of_stomach_normal**: TP=187 FP=18 TN=366 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 8 | 154 | 14 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.9506    0.8750    0.9112       176
body_of_stomach_normal     0.9122    0.9740    0.9421       192

              accuracy                         0.9323       576
             macro avg     0.9335    0.9304    0.9311       576
          weighted avg     0.9332    0.9323    0.9319       576

```
# deit_tiny_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_tiny_patch16_224_seed42/deit_tiny_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9732 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9789 |
| Balanced Acc | 0.9721 |
| Macro-Sensitivity | 0.9721 |
| Macro-Specificity | 0.9890 |
| Macro-Precision | 0.9744 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 5.49M |
| Pure-Inference FPS | 3174.6 (0.31 ms/张) |
| End-to-End FPS | 245.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9784 | 0.9745 | 0.9770 | 0.9827 | 0.9973 | 0.9970 | 195 |
| Malignant_lesions | 0.9367 | 0.9885 | 0.9487 | 0.9427 | 0.9857 | 0.9956 | 0.9793 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=5 TN=227 FN=4
- **Malignant_lesions**: TP=74 FP=4 TN=344 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9745    0.9795    0.9770       195
     Malignant_lesions     0.9487    0.9367    0.9427        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9789       427
             macro avg     0.9744    0.9721    0.9732       427
          weighted avg     0.9789    0.9789    0.9789       427

```
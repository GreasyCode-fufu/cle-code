# convnext_v2cam2 评估报告

- 权重: `weights_cle_ac_v1_c/convnext_v2cam2_seed43/convnext_v2cam2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9668 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9671 |
| Balanced Acc | 0.9668 |
| Macro-Sensitivity | 0.9668 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9669 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9504 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 372.4 (2.69 ms/张) |
| End-to-End FPS | 16.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9783 | 0.9617 | 0.9640 | 0.9809 | 0.9964 | 0.9941 | 208 |
| Malignant_lesions | 0.9602 | 0.9775 | 0.9494 | 0.9548 | 0.9824 | 0.9946 | 0.9881 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=8 TN=360 FN=7
- **Malignant_lesions**: TP=169 FP=9 TN=391 FN=7
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 6 | 169 | 1 |
| body_of_stomach_normal | 2 | 3 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9617    0.9663    0.9640       208
     Malignant_lesions     0.9494    0.9602    0.9548       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9670       576
             macro avg     0.9669    0.9668    0.9668       576
          weighted avg     0.9672    0.9670    0.9671       576

```
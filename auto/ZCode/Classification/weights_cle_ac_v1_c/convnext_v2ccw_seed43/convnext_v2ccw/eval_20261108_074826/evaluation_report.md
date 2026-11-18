# convnext_v2ccw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2ccw_seed43/convnext_v2ccw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9616 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9617 |
| Balanced Acc | 0.9627 |
| Macro-Sensitivity | 0.9627 |
| Macro-Specificity | 0.9811 |
| Macro-Precision | 0.9611 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9429 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 621.1 (1.61 ms/张) |
| End-to-End FPS | 203.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9864 | 0.9749 | 0.9533 | 0.9629 | 0.9925 | 0.9885 | 208 |
| Malignant_lesions | 0.9659 | 0.9700 | 0.9341 | 0.9497 | 0.9848 | 0.9925 | 0.9832 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9994 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=5 TN=363 FN=14
- **Malignant_lesions**: TP=170 FP=12 TN=388 FN=6
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 3 | 170 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9749    0.9327    0.9533       208
     Malignant_lesions     0.9341    0.9659    0.9497       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9618       576
             macro avg     0.9611    0.9627    0.9616       576
          weighted avg     0.9622    0.9618    0.9617       576

```
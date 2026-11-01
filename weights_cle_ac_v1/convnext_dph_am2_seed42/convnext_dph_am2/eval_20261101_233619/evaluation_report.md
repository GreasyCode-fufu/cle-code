# convnext_dph_am2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am2_seed42/convnext_dph_am2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9817 |
| Accuracy | 0.9859 |
| Macro-AUROC (OvR) | 0.9994 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9858 |
| Balanced Acc | 0.9772 |
| Macro-Sensitivity | 0.9772 |
| Macro-Specificity | 0.9921 |
| Macro-Precision | 0.9867 |
| AUC-PR (Macro) | 0.9979 |
| Cohen's Kappa | 0.9776 |
| MCC | 0.9777 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 957.0 (1.04 ms/张) |
| End-to-End FPS | 228.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9828 | 0.9798 | 0.9873 | 0.9956 | 0.9994 | 0.9993 | 195 |
| Malignant_lesions | 0.9367 | 0.9971 | 0.9867 | 0.9610 | 0.9858 | 0.9988 | 0.9944 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=4 TN=228 FN=1
- **Malignant_lesions**: TP=74 FP=1 TN=347 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9798    0.9949    0.9873       195
     Malignant_lesions     0.9867    0.9367    0.9610        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9859       427
             macro avg     0.9867    0.9772    0.9817       427
          weighted avg     0.9860    0.9859    0.9858       427

```
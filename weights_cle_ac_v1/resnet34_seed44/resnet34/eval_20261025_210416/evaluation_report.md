# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet34_seed44/resnet34/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9762 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9814 |
| Balanced Acc | 0.9783 |
| Macro-Sensitivity | 0.9783 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9743 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9703 |
| MCC | 0.9703 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 2469.8 (0.40 ms/张) |
| End-to-End FPS | 258.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9871 | 0.9845 | 0.9820 | 0.9828 | 0.9968 | 0.9959 | 195 |
| Malignant_lesions | 0.9620 | 0.9856 | 0.9383 | 0.9500 | 0.9913 | 0.9938 | 0.9759 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=3 TN=229 FN=4
- **Malignant_lesions**: TP=76 FP=5 TN=343 FN=3
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9795    0.9820       195
     Malignant_lesions     0.9383    0.9620    0.9500        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9743    0.9783    0.9762       427
          weighted avg     0.9815    0.9813    0.9814       427

```
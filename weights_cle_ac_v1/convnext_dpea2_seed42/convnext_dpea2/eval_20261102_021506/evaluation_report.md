# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dpea2_seed42/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9514 |
| Balanced Acc | 0.9528 |
| Macro-Sensitivity | 0.9528 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.9509 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9270 |
| MCC | 0.9277 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 661.6 (1.51 ms/张) |
| End-to-End FPS | 204.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9087 | 0.9837 | 0.9692 | 0.9380 | 0.9501 | 0.9939 | 0.9901 | 208 |
| Malignant_lesions | 0.9602 | 0.9525 | 0.8989 | 0.9286 | 0.9820 | 0.9933 | 0.9846 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=6 TN=362 FN=19
- **Malignant_lesions**: TP=169 FP=19 TN=381 FN=7
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 18 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9692    0.9087    0.9380       208
     Malignant_lesions     0.8989    0.9602    0.9286       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9514       576
             macro avg     0.9509    0.9528    0.9512       576
          weighted avg     0.9528    0.9514    0.9514       576

```
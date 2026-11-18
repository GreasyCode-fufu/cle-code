# convnext_v6c2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6c2_seed42/convnext_v6c2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9565 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9568 |
| Balanced Acc | 0.9572 |
| Macro-Sensitivity | 0.9572 |
| Macro-Specificity | 0.9785 |
| Macro-Precision | 0.9562 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.78G |
| FLOPs | 11.57G |
| Params | 30.61M |
| Pure-Inference FPS | 569.6 (1.76 ms/张) |
| End-to-End FPS | 141.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9783 | 0.9606 | 0.9489 | 0.9651 | 0.9946 | 0.9908 | 208 |
| Malignant_lesions | 0.9602 | 0.9625 | 0.9185 | 0.9389 | 0.9821 | 0.9954 | 0.9896 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=8 TN=360 FN=13
- **Malignant_lesions**: TP=169 FP=15 TN=385 FN=7
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 13 | 0 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 3 | 2 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9606    0.9375    0.9489       208
     Malignant_lesions     0.9185    0.9602    0.9389       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9566       576
             macro avg     0.9562    0.9572    0.9565       576
          weighted avg     0.9573    0.9566    0.9568       576

```
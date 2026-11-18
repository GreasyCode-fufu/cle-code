# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed43/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9298 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9863 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9300 |
| Balanced Acc | 0.9299 |
| Macro-Sensitivity | 0.9299 |
| Macro-Specificity | 0.9651 |
| Macro-Precision | 0.9308 |
| AUC-PR (Macro) | 0.9724 |
| Cohen's Kappa | 0.8956 |
| MCC | 0.8961 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3854.3 (0.26 ms/张) |
| End-to-End FPS | 123.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9087 | 0.9592 | 0.9265 | 0.9175 | 0.9489 | 0.9796 | 0.9697 | 208 |
| Malignant_lesions | 0.8864 | 0.9725 | 0.9341 | 0.9096 | 0.9511 | 0.9806 | 0.9503 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9635 | 0.9317 | 0.9622 | 0.9973 | 0.9987 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=15 TN=353 FN=19
- **Malignant_lesions**: TP=156 FP=11 TN=389 FN=20
- **body_of_stomach_normal**: TP=191 FP=14 TN=370 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 10 | 9 |
| Malignant_lesions | 15 | 156 | 5 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9265    0.9087    0.9175       208
     Malignant_lesions     0.9341    0.8864    0.9096       176
body_of_stomach_normal     0.9317    0.9948    0.9622       192

              accuracy                         0.9306       576
             macro avg     0.9308    0.9299    0.9298       576
          weighted avg     0.9306    0.9306    0.9300       576

```
# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed46/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9348 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9353 |
| Balanced Acc | 0.9337 |
| Macro-Sensitivity | 0.9337 |
| Macro-Specificity | 0.9674 |
| Macro-Precision | 0.9374 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9033 |
| MCC | 0.9039 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4942.1 (0.20 ms/张) |
| End-to-End FPS | 90.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9429 | 0.9032 | 0.9224 | 0.9666 | 0.9892 | 0.9815 | 208 |
| Malignant_lesions | 0.8693 | 0.9775 | 0.9444 | 0.9053 | 0.9444 | 0.9906 | 0.9818 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=21 TN=347 FN=12
- **Malignant_lesions**: TP=153 FP=9 TN=391 FN=23
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 20 | 153 | 3 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9032    0.9423    0.9224       208
     Malignant_lesions     0.9444    0.8693    0.9053       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9358       576
             macro avg     0.9374    0.9337    0.9348       576
          weighted avg     0.9362    0.9358    0.9353       576

```
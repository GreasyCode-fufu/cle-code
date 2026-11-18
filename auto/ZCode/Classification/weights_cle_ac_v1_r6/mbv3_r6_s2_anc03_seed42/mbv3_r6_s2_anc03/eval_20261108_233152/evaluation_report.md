# mbv3_r6_s2_anc03 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_anc03_seed42/mbv3_r6_s2_anc03/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9619 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9698 |
| Balanced Acc | 0.9677 |
| Macro-Sensitivity | 0.9677 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9568 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9518 |
| MCC | 0.9521 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 2188.9 (0.46 ms/张) |
| End-to-End FPS | 107.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9871 | 0.9841 | 0.9688 | 0.9622 | 0.9976 | 0.9970 | 195 |
| Malignant_lesions | 0.9494 | 0.9741 | 0.8929 | 0.9202 | 0.9883 | 0.9955 | 0.9827 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=3 TN=229 FN=9
- **Malignant_lesions**: TP=75 FP=9 TN=339 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 3 | 75 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9841    0.9538    0.9688       195
     Malignant_lesions     0.8929    0.9494    0.9202        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9696       427
             macro avg     0.9568    0.9677    0.9619       427
          weighted avg     0.9706    0.9696    0.9698       427

```
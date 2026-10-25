# convnext_wds_wtb_k3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_seed42/convnext_wds_wtb_k3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9821 |
| Accuracy | 0.9859 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9859 |
| Balanced Acc | 0.9797 |
| Macro-Sensitivity | 0.9797 |
| Macro-Specificity | 0.9923 |
| Macro-Precision | 0.9846 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9776 |
| MCC | 0.9776 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.50G |
| FLOPs | 9.01G |
| Params | 28.28M |
| Pure-Inference FPS | 1425.1 (0.70 ms/张) |
| End-to-End FPS | 261.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9828 | 0.9797 | 0.9847 | 0.9913 | 0.9981 | 0.9979 | 195 |
| Malignant_lesions | 0.9494 | 0.9943 | 0.9740 | 0.9615 | 0.9886 | 0.9970 | 0.9874 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=4 TN=228 FN=2
- **Malignant_lesions**: TP=75 FP=2 TN=346 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9797    0.9897    0.9847       195
     Malignant_lesions     0.9740    0.9494    0.9615        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9859       427
             macro avg     0.9846    0.9797    0.9821       427
          weighted avg     0.9859    0.9859    0.9859       427

```
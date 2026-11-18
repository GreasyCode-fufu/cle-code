# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9707 |
| Accuracy | 0.9705 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9704 |
| Balanced Acc | 0.9710 |
| Macro-Sensitivity | 0.9710 |
| Macro-Specificity | 0.9852 |
| Macro-Precision | 0.9703 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9557 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5642.4 (0.18 ms/张) |
| End-to-End FPS | 234.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9810 | 0.9659 | 0.9588 | 0.9730 | 0.9938 | 0.9912 | 208 |
| Malignant_lesions | 0.9716 | 0.9825 | 0.9607 | 0.9661 | 0.9874 | 0.9959 | 0.9888 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=7 TN=361 FN=10
- **Malignant_lesions**: TP=171 FP=7 TN=393 FN=5
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 5 | 171 | 0 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9659    0.9519    0.9588       208
     Malignant_lesions     0.9607    0.9716    0.9661       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9705       576
             macro avg     0.9703    0.9710    0.9707       576
          weighted avg     0.9705    0.9705    0.9704       576

```
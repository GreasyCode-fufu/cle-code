# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9604 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9675 |
| Balanced Acc | 0.9685 |
| Macro-Sensitivity | 0.9685 |
| Macro-Specificity | 0.9849 |
| Macro-Precision | 0.9537 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9482 |
| MCC | 0.9487 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3166.4 (0.32 ms/张) |
| End-to-End FPS | 16.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9436 | 0.9871 | 0.9840 | 0.9634 | 0.9542 | 0.9968 | 0.9962 | 195 |
| Malignant_lesions | 0.9620 | 0.9713 | 0.8837 | 0.9212 | 0.9912 | 0.9947 | 0.9790 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=3 TN=229 FN=11
- **Malignant_lesions**: TP=76 FP=10 TN=338 FN=3
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 10 | 1 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9840    0.9436    0.9634       195
     Malignant_lesions     0.8837    0.9620    0.9212        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9537    0.9685    0.9604       427
          weighted avg     0.9688    0.9672    0.9675       427

```
# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9447 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9931 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9446 |
| Balanced Acc | 0.9431 |
| Macro-Sensitivity | 0.9431 |
| Macro-Specificity | 0.9716 |
| Macro-Precision | 0.9474 |
| AUC-PR (Macro) | 0.9870 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9169 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4218.1 (0.24 ms/张) |
| End-to-End FPS | 15.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9429 | 0.9045 | 0.9299 | 0.9747 | 0.9884 | 0.9805 | 208 |
| Malignant_lesions | 0.9091 | 0.9850 | 0.9639 | 0.9357 | 0.9610 | 0.9926 | 0.9837 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9870 | 0.9737 | 0.9686 | 0.9819 | 0.9983 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=21 TN=347 FN=9
- **Malignant_lesions**: TP=160 FP=6 TN=394 FN=16
- **body_of_stomach_normal**: TP=185 FP=5 TN=379 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 14 | 160 | 2 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9045    0.9567    0.9299       208
     Malignant_lesions     0.9639    0.9091    0.9357       176
body_of_stomach_normal     0.9737    0.9635    0.9686       192

              accuracy                         0.9444       576
             macro avg     0.9474    0.9431    0.9447       576
          weighted avg     0.9457    0.9444    0.9446       576

```
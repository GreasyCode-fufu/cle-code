# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9586 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9673 |
| Balanced Acc | 0.9610 |
| Macro-Sensitivity | 0.9610 |
| Macro-Specificity | 0.9839 |
| Macro-Precision | 0.9562 |
| AUC-PR (Macro) | 0.9859 |
| Cohen's Kappa | 0.9480 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3136.6 (0.32 ms/张) |
| End-to-End FPS | 14.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9784 | 0.9740 | 0.9664 | 0.9660 | 0.9950 | 0.9941 | 195 |
| Malignant_lesions | 0.9241 | 0.9770 | 0.9012 | 0.9125 | 0.9827 | 0.9910 | 0.9635 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=5 TN=227 FN=8
- **Malignant_lesions**: TP=73 FP=8 TN=340 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9740    0.9590    0.9664       195
     Malignant_lesions     0.9012    0.9241    0.9125        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9562    0.9610    0.9586       427
          weighted avg     0.9675    0.9672    0.9673       427

```
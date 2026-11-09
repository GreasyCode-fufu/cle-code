# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed42/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9156 |
| Accuracy | 0.9167 |
| Macro-AUROC (OvR) | 0.9905 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9161 |
| Balanced Acc | 0.9122 |
| Macro-Sensitivity | 0.9122 |
| Macro-Specificity | 0.9571 |
| Macro-Precision | 0.9270 |
| AUC-PR (Macro) | 0.9821 |
| Cohen's Kappa | 0.8742 |
| MCC | 0.8778 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2465.6 (0.41 ms/张) |
| End-to-End FPS | 80.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9022 | 0.8481 | 0.9034 | 0.9794 | 0.9859 | 0.9747 | 208 |
| Malignant_lesions | 0.8068 | 0.9925 | 0.9793 | 0.8847 | 0.9211 | 0.9877 | 0.9759 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9766 | 0.9536 | 0.9585 | 0.9817 | 0.9978 | 0.9958 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=36 TN=332 FN=7
- **Malignant_lesions**: TP=142 FP=3 TN=397 FN=34
- **body_of_stomach_normal**: TP=185 FP=9 TN=375 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 3 | 4 |
| Malignant_lesions | 29 | 142 | 5 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8481    0.9663    0.9034       208
     Malignant_lesions     0.9793    0.8068    0.8847       176
body_of_stomach_normal     0.9536    0.9635    0.9585       192

              accuracy                         0.9167       576
             macro avg     0.9270    0.9122    0.9156       576
          weighted avg     0.9234    0.9167    0.9161       576

```
# starnet_s2_lp_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_lp_dce4s_seed42/starnet_s2_lp_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9305 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9303 |
| Balanced Acc | 0.9297 |
| Macro-Sensitivity | 0.9297 |
| Macro-Specificity | 0.9648 |
| Macro-Precision | 0.9324 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.8955 |
| MCC | 0.8960 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3668.5 (0.27 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9511 | 0.9143 | 0.9187 | 0.9563 | 0.9857 | 0.9786 | 208 |
| Malignant_lesions | 0.8920 | 0.9825 | 0.9573 | 0.9235 | 0.9539 | 0.9931 | 0.9854 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9609 | 0.9257 | 0.9492 | 0.9866 | 0.9960 | 0.9918 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=18 TN=350 FN=16
- **Malignant_lesions**: TP=157 FP=7 TN=393 FN=19
- **body_of_stomach_normal**: TP=187 FP=15 TN=369 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 7 | 9 |
| Malignant_lesions | 13 | 157 | 6 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9143    0.9231    0.9187       208
     Malignant_lesions     0.9573    0.8920    0.9235       176
body_of_stomach_normal     0.9257    0.9740    0.9492       192

              accuracy                         0.9306       576
             macro avg     0.9324    0.9297    0.9305       576
          weighted avg     0.9313    0.9306    0.9303       576

```
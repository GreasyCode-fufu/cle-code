# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_s_seed44/lsnet_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9081 |
| Accuracy | 0.9097 |
| Macro-AUROC (OvR) | 0.9885 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9088 |
| Balanced Acc | 0.9060 |
| Macro-Sensitivity | 0.9060 |
| Macro-Specificity | 0.9539 |
| Macro-Precision | 0.9158 |
| AUC-PR (Macro) | 0.9788 |
| Cohen's Kappa | 0.8638 |
| MCC | 0.8661 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 3617.1 (0.28 ms/张) |
| End-to-End FPS | 196.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9130 | 0.8596 | 0.8991 | 0.9655 | 0.9824 | 0.9714 | 208 |
| Malignant_lesions | 0.8068 | 0.9825 | 0.9530 | 0.8738 | 0.9204 | 0.9865 | 0.9714 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9661 | 0.9347 | 0.9514 | 0.9841 | 0.9965 | 0.9935 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=32 TN=336 FN=12
- **Malignant_lesions**: TP=142 FP=7 TN=393 FN=34
- **body_of_stomach_normal**: TP=186 FP=13 TN=371 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 7 | 5 |
| Malignant_lesions | 26 | 142 | 8 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8596    0.9423    0.8991       208
     Malignant_lesions     0.9530    0.8068    0.8738       176
body_of_stomach_normal     0.9347    0.9688    0.9514       192

              accuracy                         0.9097       576
             macro avg     0.9158    0.9060    0.9081       576
          weighted avg     0.9132    0.9097    0.9088       576

```
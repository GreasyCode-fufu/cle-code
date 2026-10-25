# convnext_wds_wtb_l1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_l1_seed42/convnext_wds_wtb_l1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9514 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9513 |
| Balanced Acc | 0.9527 |
| Macro-Sensitivity | 0.9527 |
| Macro-Specificity | 0.9759 |
| Macro-Precision | 0.9508 |
| AUC-PR (Macro) | 0.9876 |
| Cohen's Kappa | 0.9270 |
| MCC | 0.9274 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.06G |
| Params | 28.53M |
| Pure-Inference FPS | 534.6 (1.87 ms/张) |
| End-to-End FPS | 192.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9783 | 0.9596 | 0.9360 | 0.9524 | 0.9915 | 0.9851 | 208 |
| Malignant_lesions | 0.9602 | 0.9600 | 0.9135 | 0.9363 | 0.9821 | 0.9887 | 0.9803 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9988 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=8 TN=360 FN=18
- **Malignant_lesions**: TP=169 FP=16 TN=384 FN=7
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 16 | 2 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9596    0.9135    0.9360       208
     Malignant_lesions     0.9135    0.9602    0.9363       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9514       576
             macro avg     0.9508    0.9527    0.9514       576
          weighted avg     0.9521    0.9514    0.9513       576

```
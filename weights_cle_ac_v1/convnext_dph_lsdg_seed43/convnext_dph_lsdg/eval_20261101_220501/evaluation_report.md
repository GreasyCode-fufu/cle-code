# convnext_dph_lsdg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_lsdg_seed43/convnext_dph_lsdg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9565 |
| Balanced Acc | 0.9573 |
| Macro-Sensitivity | 0.9573 |
| Macro-Specificity | 0.9786 |
| Macro-Precision | 0.9559 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9354 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.27G |
| Params | 28.93M |
| Pure-Inference FPS | 537.5 (1.86 ms/张) |
| End-to-End FPS | 201.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9918 | 0.9846 | 0.9529 | 0.9580 | 0.9943 | 0.9923 | 208 |
| Malignant_lesions | 0.9489 | 0.9675 | 0.9278 | 0.9382 | 0.9773 | 0.9922 | 0.9847 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9766 | 0.9552 | 0.9771 | 1.0000 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=3 TN=365 FN=16
- **Malignant_lesions**: TP=167 FP=13 TN=387 FN=9
- **body_of_stomach_normal**: TP=192 FP=9 TN=375 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 13 | 3 |
| Malignant_lesions | 3 | 167 | 6 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9846    0.9231    0.9529       208
     Malignant_lesions     0.9278    0.9489    0.9382       176
body_of_stomach_normal     0.9552    1.0000    0.9771       192

              accuracy                         0.9566       576
             macro avg     0.9559    0.9573    0.9561       576
          weighted avg     0.9575    0.9566    0.9565       576

```
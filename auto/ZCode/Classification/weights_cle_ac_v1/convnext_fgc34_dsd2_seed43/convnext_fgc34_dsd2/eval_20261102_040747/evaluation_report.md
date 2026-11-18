# convnext_fgc34_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc34_dsd2_seed43/convnext_fgc34_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9249 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9887 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9253 |
| Balanced Acc | 0.9236 |
| Macro-Sensitivity | 0.9236 |
| Macro-Specificity | 0.9622 |
| Macro-Precision | 0.9270 |
| AUC-PR (Macro) | 0.9793 |
| Cohen's Kappa | 0.8876 |
| MCC | 0.8880 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.40G |
| FLOPs | 8.80G |
| Params | 27.49M |
| Pure-Inference FPS | 1722.6 (0.58 ms/张) |
| End-to-End FPS | 268.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9321 | 0.8858 | 0.9087 | 0.9608 | 0.9833 | 0.9717 | 208 |
| Malignant_lesions | 0.8693 | 0.9650 | 0.9162 | 0.8921 | 0.9438 | 0.9868 | 0.9725 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9960 | 0.9936 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=25 TN=343 FN=14
- **Malignant_lesions**: TP=153 FP=14 TN=386 FN=23
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 20 | 153 | 3 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8858    0.9327    0.9087       208
     Malignant_lesions     0.9162    0.8693    0.8921       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9253       576
             macro avg     0.9270    0.9236    0.9249       576
          weighted avg     0.9261    0.9253    0.9253       576

```
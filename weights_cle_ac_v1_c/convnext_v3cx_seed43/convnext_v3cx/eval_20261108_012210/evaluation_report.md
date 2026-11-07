# convnext_v3cx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cx_seed43/convnext_v3cx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9472 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9951 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9476 |
| Balanced Acc | 0.9463 |
| Macro-Sensitivity | 0.9463 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9490 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9220 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 459.7 (2.18 ms/张) |
| End-to-End FPS | 183.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9592 | 0.9299 | 0.9431 | 0.9751 | 0.9935 | 0.9896 | 208 |
| Malignant_lesions | 0.8977 | 0.9825 | 0.9576 | 0.9267 | 0.9562 | 0.9927 | 0.9825 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9792 | 0.9594 | 0.9717 | 0.9921 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=15 TN=353 FN=9
- **Malignant_lesions**: TP=158 FP=7 TN=393 FN=18
- **body_of_stomach_normal**: TP=189 FP=8 TN=376 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 12 | 158 | 6 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9299    0.9567    0.9431       208
     Malignant_lesions     0.9576    0.8977    0.9267       176
body_of_stomach_normal     0.9594    0.9844    0.9717       192

              accuracy                         0.9479       576
             macro avg     0.9490    0.9463    0.9472       576
          weighted avg     0.9482    0.9479    0.9476       576

```
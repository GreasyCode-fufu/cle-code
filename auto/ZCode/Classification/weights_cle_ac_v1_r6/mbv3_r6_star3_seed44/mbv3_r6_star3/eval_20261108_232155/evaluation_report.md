# mbv3_r6_star3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star3_seed44/mbv3_r6_star3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9036 |
| Accuracy | 0.9062 |
| Macro-AUROC (OvR) | 0.9893 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9047 |
| Balanced Acc | 0.9002 |
| Macro-Sensitivity | 0.9002 |
| Macro-Specificity | 0.9516 |
| Macro-Precision | 0.9200 |
| AUC-PR (Macro) | 0.9792 |
| Cohen's Kappa | 0.8583 |
| MCC | 0.8642 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.97M |
| FLOPs | 463.94M |
| Params | 3.26M |
| Pure-Inference FPS | 2601.6 (0.38 ms/张) |
| End-to-End FPS | 139.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.8777 | 0.8171 | 0.8855 | 0.9788 | 0.9819 | 0.9700 | 208 |
| Malignant_lesions | 0.7500 | 0.9875 | 0.9635 | 0.8435 | 0.8998 | 0.9862 | 0.9681 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=45 TN=323 FN=7
- **Malignant_lesions**: TP=132 FP=5 TN=395 FN=44
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 42 | 132 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8171    0.9663    0.8855       208
     Malignant_lesions     0.9635    0.7500    0.8435       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9062       576
             macro avg     0.9200    0.9002    0.9036       576
          weighted avg     0.9159    0.9062    0.9047       576

```
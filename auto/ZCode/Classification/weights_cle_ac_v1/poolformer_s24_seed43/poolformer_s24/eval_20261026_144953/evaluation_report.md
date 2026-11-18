# poolformer_s24 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/poolformer_s24_seed43/poolformer_s24/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9544 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9528 |
| Macro-Sensitivity | 0.9528 |
| Macro-Specificity | 0.9769 |
| Macro-Precision | 0.9573 |
| AUC-PR (Macro) | 0.9897 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9327 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.39G |
| FLOPs | 6.79G |
| Params | 20.84M |
| Pure-Inference FPS | 1218.2 (0.82 ms/张) |
| End-to-End FPS | 247.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9511 | 0.9182 | 0.9439 | 0.9831 | 0.9923 | 0.9852 | 208 |
| Malignant_lesions | 0.8977 | 0.9875 | 0.9693 | 0.9322 | 0.9564 | 0.9913 | 0.9841 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=18 TN=350 FN=6
- **Malignant_lesions**: TP=158 FP=5 TN=395 FN=18
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 16 | 158 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9182    0.9712    0.9439       208
     Malignant_lesions     0.9693    0.8977    0.9322       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9549       576
             macro avg     0.9573    0.9528    0.9544       576
          weighted avg     0.9559    0.9549    0.9547       576

```
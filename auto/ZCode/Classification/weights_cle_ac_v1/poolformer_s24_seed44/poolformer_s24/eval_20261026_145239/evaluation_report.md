# poolformer_s24 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/poolformer_s24_seed44/poolformer_s24/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9359 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9360 |
| Balanced Acc | 0.9334 |
| Macro-Sensitivity | 0.9334 |
| Macro-Specificity | 0.9671 |
| Macro-Precision | 0.9409 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9032 |
| MCC | 0.9046 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.39G |
| FLOPs | 6.79G |
| Params | 20.84M |
| Pure-Inference FPS | 2089.1 (0.48 ms/张) |
| End-to-End FPS | 275.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9266 | 0.8821 | 0.9245 | 0.9827 | 0.9914 | 0.9859 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9896 | 0.9782 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9922 | 0.9837 | 0.9628 | 0.9719 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=27 TN=341 FN=6
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=181 FP=3 TN=381 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 6 | 0 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 10 | 1 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8821    0.9712    0.9245       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9837    0.9427    0.9628       192

              accuracy                         0.9358       576
             macro avg     0.9409    0.9334    0.9359       576
          weighted avg     0.9389    0.9358    0.9360       576

```
# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed44/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9273 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9274 |
| Balanced Acc | 0.9235 |
| Macro-Sensitivity | 0.9235 |
| Macro-Specificity | 0.9623 |
| Macro-Precision | 0.9376 |
| AUC-PR (Macro) | 0.9842 |
| Cohen's Kappa | 0.8899 |
| MCC | 0.8931 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 4845.9 (0.21 ms/张) |
| End-to-End FPS | 215.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9022 | 0.8487 | 0.9058 | 0.9822 | 0.9874 | 0.9803 | 208 |
| Malignant_lesions | 0.8409 | 0.9925 | 0.9801 | 0.9052 | 0.9341 | 0.9887 | 0.9781 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9971 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=36 TN=332 FN=6
- **Malignant_lesions**: TP=148 FP=3 TN=397 FN=28
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 3 | 3 |
| Malignant_lesions | 28 | 148 | 0 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8487    0.9712    0.9058       208
     Malignant_lesions     0.9801    0.8409    0.9052       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9271       576
             macro avg     0.9376    0.9235    0.9273       576
          weighted avg     0.9340    0.9271    0.9274       576

```
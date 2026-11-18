# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed45/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9362 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9361 |
| Balanced Acc | 0.9329 |
| Macro-Sensitivity | 0.9329 |
| Macro-Specificity | 0.9669 |
| Macro-Precision | 0.9436 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9031 |
| MCC | 0.9052 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1250.9 (0.80 ms/张) |
| End-to-End FPS | 221.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9158 | 0.8670 | 0.9161 | 0.9825 | 0.9878 | 0.9779 | 208 |
| Malignant_lesions | 0.8693 | 0.9900 | 0.9745 | 0.9189 | 0.9451 | 0.9879 | 0.9774 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9974 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=31 TN=337 FN=6
- **Malignant_lesions**: TP=153 FP=4 TN=396 FN=23
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 23 | 153 | 0 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8670    0.9712    0.9161       208
     Malignant_lesions     0.9745    0.8693    0.9189       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9358       576
             macro avg     0.9436    0.9329    0.9362       576
          weighted avg     0.9406    0.9358    0.9361       576

```
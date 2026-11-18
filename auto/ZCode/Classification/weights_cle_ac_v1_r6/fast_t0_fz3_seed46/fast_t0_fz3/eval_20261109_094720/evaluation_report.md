# fast_t0_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_seed46/fast_t0_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9420 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9424 |
| Balanced Acc | 0.9416 |
| Macro-Sensitivity | 0.9416 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9433 |
| AUC-PR (Macro) | 0.9865 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9142 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 5616.1 (0.18 ms/张) |
| End-to-End FPS | 191.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9647 | 0.9375 | 0.9375 | 0.9647 | 0.9896 | 0.9845 | 208 |
| Malignant_lesions | 0.8977 | 0.9800 | 0.9518 | 0.9240 | 0.9561 | 0.9913 | 0.9772 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9688 | 0.9406 | 0.9645 | 0.9947 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=13 TN=355 FN=13
- **Malignant_lesions**: TP=158 FP=8 TN=392 FN=18
- **body_of_stomach_normal**: TP=190 FP=12 TN=372 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 8 | 5 |
| Malignant_lesions | 11 | 158 | 7 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9375    0.9375    0.9375       208
     Malignant_lesions     0.9518    0.8977    0.9240       176
body_of_stomach_normal     0.9406    0.9896    0.9645       192

              accuracy                         0.9427       576
             macro avg     0.9433    0.9416    0.9420       576
          weighted avg     0.9429    0.9427    0.9424       576

```
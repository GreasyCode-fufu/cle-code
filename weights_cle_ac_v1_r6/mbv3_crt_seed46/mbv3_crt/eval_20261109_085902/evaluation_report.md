# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed46/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9412 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9880 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9413 |
| Balanced Acc | 0.9394 |
| Macro-Sensitivity | 0.9394 |
| Macro-Specificity | 0.9699 |
| Macro-Precision | 0.9446 |
| AUC-PR (Macro) | 0.9752 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9119 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 2249.9 (0.44 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9348 | 0.8929 | 0.9259 | 0.9773 | 0.9834 | 0.9676 | 208 |
| Malignant_lesions | 0.9034 | 0.9775 | 0.9464 | 0.9244 | 0.9583 | 0.9838 | 0.9625 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9974 | 0.9946 | 0.9734 | 0.9770 | 0.9970 | 0.9955 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=24 TN=344 FN=8
- **Malignant_lesions**: TP=159 FP=9 TN=391 FN=17
- **body_of_stomach_normal**: TP=183 FP=1 TN=383 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 8 | 0 |
| Malignant_lesions | 16 | 159 | 1 |
| body_of_stomach_normal | 8 | 1 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8929    0.9615    0.9259       208
     Malignant_lesions     0.9464    0.9034    0.9244       176
body_of_stomach_normal     0.9946    0.9531    0.9734       192

              accuracy                         0.9410       576
             macro avg     0.9446    0.9394    0.9412       576
          weighted avg     0.9431    0.9410    0.9413       576

```
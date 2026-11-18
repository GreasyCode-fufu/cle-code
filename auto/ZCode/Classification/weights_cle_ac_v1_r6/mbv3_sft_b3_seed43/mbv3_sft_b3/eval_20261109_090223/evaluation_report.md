# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed43/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9335 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9892 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9339 |
| Balanced Acc | 0.9316 |
| Macro-Sensitivity | 0.9316 |
| Macro-Specificity | 0.9663 |
| Macro-Precision | 0.9375 |
| AUC-PR (Macro) | 0.9808 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9016 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2725.2 (0.37 ms/张) |
| End-to-End FPS | 127.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9321 | 0.8884 | 0.9213 | 0.9744 | 0.9852 | 0.9745 | 208 |
| Malignant_lesions | 0.8693 | 0.9800 | 0.9503 | 0.9080 | 0.9446 | 0.9855 | 0.9736 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9870 | 0.9738 | 0.9713 | 0.9844 | 0.9971 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=25 TN=343 FN=9
- **Malignant_lesions**: TP=153 FP=8 TN=392 FN=23
- **body_of_stomach_normal**: TP=186 FP=5 TN=379 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 20 | 153 | 3 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8884    0.9567    0.9213       208
     Malignant_lesions     0.9503    0.8693    0.9080       176
body_of_stomach_normal     0.9738    0.9688    0.9713       192

              accuracy                         0.9340       576
             macro avg     0.9375    0.9316    0.9335       576
          weighted avg     0.9358    0.9340    0.9339       576

```
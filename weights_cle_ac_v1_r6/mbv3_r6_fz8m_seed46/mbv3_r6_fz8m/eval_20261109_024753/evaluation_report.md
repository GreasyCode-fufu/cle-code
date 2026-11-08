# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed46/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9471 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9475 |
| Balanced Acc | 0.9451 |
| Macro-Sensitivity | 0.9451 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.9523 |
| AUC-PR (Macro) | 0.9869 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9229 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3128.0 (0.32 ms/张) |
| End-to-End FPS | 244.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9402 | 0.9018 | 0.9352 | 0.9830 | 0.9897 | 0.9805 | 208 |
| Malignant_lesions | 0.8693 | 0.9925 | 0.9808 | 0.9217 | 0.9452 | 0.9896 | 0.9808 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=22 TN=346 FN=6
- **Malignant_lesions**: TP=153 FP=3 TN=397 FN=23
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 3 | 3 |
| Malignant_lesions | 21 | 153 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9018    0.9712    0.9352       208
     Malignant_lesions     0.9808    0.8693    0.9217       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9479       576
             macro avg     0.9523    0.9451    0.9471       576
          weighted avg     0.9502    0.9479    0.9475       576

```
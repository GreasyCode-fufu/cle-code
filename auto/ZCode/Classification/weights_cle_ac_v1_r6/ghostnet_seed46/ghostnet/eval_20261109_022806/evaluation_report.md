# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/ghostnet_seed46/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9337 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9875 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9340 |
| Balanced Acc | 0.9338 |
| Macro-Sensitivity | 0.9338 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.9336 |
| AUC-PR (Macro) | 0.9766 |
| Cohen's Kappa | 0.9008 |
| MCC | 0.9008 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 4457.5 (0.22 ms/张) |
| End-to-End FPS | 104.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9592 | 0.9275 | 0.9253 | 0.9566 | 0.9822 | 0.9699 | 208 |
| Malignant_lesions | 0.9148 | 0.9600 | 0.9096 | 0.9122 | 0.9624 | 0.9824 | 0.9643 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9818 | 0.9635 | 0.9635 | 0.9818 | 0.9978 | 0.9955 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=15 TN=353 FN=16
- **Malignant_lesions**: TP=161 FP=16 TN=384 FN=15
- **body_of_stomach_normal**: TP=185 FP=7 TN=377 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 3 | 4 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9275    0.9231    0.9253       208
     Malignant_lesions     0.9096    0.9148    0.9122       176
body_of_stomach_normal     0.9635    0.9635    0.9635       192

              accuracy                         0.9340       576
             macro avg     0.9336    0.9338    0.9337       576
          weighted avg     0.9341    0.9340    0.9340       576

```
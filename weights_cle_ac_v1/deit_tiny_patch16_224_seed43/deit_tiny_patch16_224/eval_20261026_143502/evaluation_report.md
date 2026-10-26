# deit_tiny_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_tiny_patch16_224_seed43/deit_tiny_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9451 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9449 |
| Balanced Acc | 0.9423 |
| Macro-Sensitivity | 0.9423 |
| Macro-Specificity | 0.9713 |
| Macro-Precision | 0.9510 |
| AUC-PR (Macro) | 0.9920 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9179 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 5.49M |
| Pure-Inference FPS | 4235.1 (0.24 ms/张) |
| End-to-End FPS | 254.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9266 | 0.8831 | 0.9294 | 0.9884 | 0.9933 | 0.9874 | 208 |
| Malignant_lesions | 0.9034 | 0.9900 | 0.9755 | 0.9381 | 0.9588 | 0.9948 | 0.9899 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9974 | 0.9945 | 0.9679 | 0.9721 | 0.9994 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=27 TN=341 FN=4
- **Malignant_lesions**: TP=159 FP=4 TN=396 FN=17
- **body_of_stomach_normal**: TP=181 FP=1 TN=383 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 3 | 1 |
| Malignant_lesions | 17 | 159 | 0 |
| body_of_stomach_normal | 10 | 1 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8831    0.9808    0.9294       208
     Malignant_lesions     0.9755    0.9034    0.9381       176
body_of_stomach_normal     0.9945    0.9427    0.9679       192

              accuracy                         0.9444       576
             macro avg     0.9510    0.9423    0.9451       576
          weighted avg     0.9485    0.9444    0.9449       576

```
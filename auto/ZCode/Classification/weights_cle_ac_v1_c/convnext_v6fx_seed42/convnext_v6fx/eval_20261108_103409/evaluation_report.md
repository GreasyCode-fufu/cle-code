# convnext_v6fx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6fx_seed42/convnext_v6fx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9559 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9553 |
| Macro-Sensitivity | 0.9553 |
| Macro-Specificity | 0.9781 |
| Macro-Precision | 0.9572 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.70M |
| Pure-Inference FPS | 459.0 (2.18 ms/张) |
| End-to-End FPS | 184.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9701 | 0.9479 | 0.9547 | 0.9781 | 0.9943 | 0.9923 | 208 |
| Malignant_lesions | 0.9148 | 0.9850 | 0.9641 | 0.9388 | 0.9633 | 0.9963 | 0.9922 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9980 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=11 TN=357 FN=8
- **Malignant_lesions**: TP=161 FP=6 TN=394 FN=15
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 9 | 161 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9479    0.9615    0.9547       208
     Malignant_lesions     0.9641    0.9148    0.9388       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9566       576
             macro avg     0.9572    0.9553    0.9559       576
          weighted avg     0.9567    0.9566    0.9564       576

```
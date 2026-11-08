# convnext_v2cam 评估报告

- 权重: `weights_cle_ac_v1_c/convnext_v2cam_seed43/convnext_v2cam/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9562 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9559 |
| Macro-Sensitivity | 0.9559 |
| Macro-Specificity | 0.9782 |
| Macro-Precision | 0.9568 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 364.3 (2.75 ms/张) |
| End-to-End FPS | 16.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9728 | 0.9519 | 0.9519 | 0.9728 | 0.9937 | 0.9901 | 208 |
| Malignant_lesions | 0.9261 | 0.9825 | 0.9588 | 0.9422 | 0.9680 | 0.9930 | 0.9830 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9993 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=10 TN=358 FN=10
- **Malignant_lesions**: TP=163 FP=7 TN=393 FN=13
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 8 | 163 | 5 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9519    0.9519    0.9519       208
     Malignant_lesions     0.9588    0.9261    0.9422       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9566       576
             macro avg     0.9568    0.9559    0.9562       576
          weighted avg     0.9566    0.9566    0.9564       576

```
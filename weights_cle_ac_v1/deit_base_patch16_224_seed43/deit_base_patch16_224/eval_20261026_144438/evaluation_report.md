# deit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_base_patch16_224_seed43/deit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9381 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9388 |
| Balanced Acc | 0.9374 |
| Macro-Sensitivity | 0.9374 |
| Macro-Specificity | 0.9693 |
| Macro-Precision | 0.9402 |
| AUC-PR (Macro) | 0.9877 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9091 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 362.7 (2.76 ms/张) |
| End-to-End FPS | 164.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9917 | 0.9876 | 208 |
| Malignant_lesions | 0.8807 | 0.9800 | 0.9509 | 0.9145 | 0.9492 | 0.9917 | 0.9796 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9688 | 0.9403 | 0.9618 | 0.9920 | 0.9978 | 0.9960 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=155 FP=8 TN=392 FN=21
- **body_of_stomach_normal**: TP=189 FP=12 TN=372 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 12 | 155 | 9 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9509    0.8807    0.9145       176
body_of_stomach_normal     0.9403    0.9844    0.9618       192

              accuracy                         0.9392       576
             macro avg     0.9402    0.9374    0.9381       576
          weighted avg     0.9396    0.9392    0.9388       576

```
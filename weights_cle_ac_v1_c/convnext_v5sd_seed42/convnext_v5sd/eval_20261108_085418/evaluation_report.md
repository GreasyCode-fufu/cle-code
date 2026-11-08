# convnext_v5sd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5sd_seed42/convnext_v5sd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9634 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9626 |
| Macro-Sensitivity | 0.9626 |
| Macro-Specificity | 0.9815 |
| Macro-Precision | 0.9645 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9451 |
| MCC | 0.9453 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 769.8 (1.30 ms/张) |
| End-to-End FPS | 225.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9647 | 0.9393 | 0.9526 | 0.9807 | 0.9951 | 0.9919 | 208 |
| Malignant_lesions | 0.9318 | 0.9850 | 0.9647 | 0.9480 | 0.9704 | 0.9953 | 0.9892 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=13 TN=355 FN=7
- **Malignant_lesions**: TP=164 FP=6 TN=394 FN=12
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9393    0.9663    0.9526       208
     Malignant_lesions     0.9647    0.9318    0.9480       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9635       576
             macro avg     0.9645    0.9626    0.9634       576
          weighted avg     0.9638    0.9635    0.9635       576

```
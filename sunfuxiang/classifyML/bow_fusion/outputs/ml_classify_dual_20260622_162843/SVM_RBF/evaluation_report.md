# SVM-RBF 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8919 |
| Accuracy | 0.8912 |
| Macro-AUROC (OvR) | 0.9749 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8904 |
| Balanced Acc | 0.8944 |
| Macro-Sensitivity | 0.8944 |
| Macro-Specificity | 0.9453 |
| Macro-Precision | 0.8910 |
| AUC-PR (Macro) | 0.9529 |
| Cohen's Kappa | 0.8363 |
| MCC | 0.8370 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 57305.3 (0.02 ms/张) |
| End-to-End FPS | 784.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9373 | 0.8939 | 0.8743 | 0.9131 | 0.9679 | 0.9518 | 256 |
| Malignant_lesions | 0.8571 | 0.9501 | 0.8867 | 0.8717 | 0.9359 | 0.9640 | 0.9344 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9485 | 0.8924 | 0.9299 | 0.9866 | 0.9928 | 0.9726 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=26 TN=389 FN=37
- **Malignant_lesions**: TP=180 FP=23 TN=438 FN=30
- **body_of_stomach_normal**: TP=199 FP=24 TN=442 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 21 | 16 |
| Malignant_lesions | 22 | 180 | 8 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8939    0.8555    0.8743       256
     Malignant_lesions     0.8867    0.8571    0.8717       210
body_of_stomach_normal     0.8924    0.9707    0.9299       205

              accuracy                         0.8912       671
             macro avg     0.8910    0.8944    0.8919       671
          weighted avg     0.8912    0.8912    0.8904       671

```
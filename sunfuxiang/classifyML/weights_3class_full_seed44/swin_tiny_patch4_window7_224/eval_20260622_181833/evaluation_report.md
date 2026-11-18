# swin_tiny_patch4_window7_224 评估报告

- 权重: `weights_3class_full_seed44/swin_tiny_patch4_window7_224/swin_tiny_patch4_window7_224_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8713 |
| Accuracy | 0.8703 |
| Macro-AUROC (OvR) | 0.9714 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8693 |
| Balanced Acc | 0.8693 |
| Macro-Sensitivity | 0.8693 |
| Macro-Specificity | 0.9329 |
| Macro-Precision | 0.8776 |
| AUC-PR (Macro) | 0.9502 |
| Cohen's Kappa | 0.8037 |
| MCC | 0.8058 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.92G |
| Params | 27.94M |
| Pure-Inference FPS | 1526.7 (0.66 ms/张) |
| End-to-End FPS | 239.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8945 | 0.8723 | 0.8121 | 0.8513 | 0.9306 | 0.9602 | 0.9430 | 256 |
| Malignant_lesions | 0.7571 | 0.9523 | 0.8785 | 0.8133 | 0.8959 | 0.9629 | 0.9256 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9742 | 0.9423 | 0.9492 | 0.9806 | 0.9912 | 0.9818 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=229 FP=53 TN=362 FN=27
- **Malignant_lesions**: TP=159 FP=22 TN=439 FN=51
- **body_of_stomach_normal**: TP=196 FP=12 TN=454 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 229 | 21 | 6 |
| Malignant_lesions | 45 | 159 | 6 |
| body_of_stomach_normal | 8 | 1 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8121    0.8945    0.8513       256
     Malignant_lesions     0.8785    0.7571    0.8133       210
body_of_stomach_normal     0.9423    0.9561    0.9492       205

              accuracy                         0.8703       671
             macro avg     0.8776    0.8693    0.8713       671
          weighted avg     0.8726    0.8703    0.8693       671

```
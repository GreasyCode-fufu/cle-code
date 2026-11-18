# repvit_m1 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed42/repvit_m1/repvit_m1_3class_hq_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8665 |
| Accuracy | 0.8659 |
| Macro-AUROC (OvR) | 0.9634 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8652 |
| Balanced Acc | 0.8707 |
| Macro-Sensitivity | 0.8707 |
| Macro-Specificity | 0.9333 |
| Macro-Precision | 0.8652 |
| AUC-PR (Macro) | 0.9330 |
| Cohen's Kappa | 0.7986 |
| MCC | 0.8001 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3864.6 (0.26 ms/张) |
| End-to-End FPS | 216.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8086 | 0.9446 | 0.9000 | 0.8519 | 0.8889 | 0.9505 | 0.9337 | 256 |
| Malignant_lesions | 0.8619 | 0.9219 | 0.8341 | 0.8478 | 0.9361 | 0.9539 | 0.8959 | 210 |
| body_of_stomach_normal | 0.9415 | 0.9335 | 0.8616 | 0.8998 | 0.9732 | 0.9858 | 0.9693 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=207 FP=23 TN=392 FN=49
- **Malignant_lesions**: TP=181 FP=36 TN=425 FN=29
- **body_of_stomach_normal**: TP=193 FP=31 TN=435 FN=12

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 207 | 30 | 19 |
| Malignant_lesions | 17 | 181 | 12 |
| body_of_stomach_normal | 6 | 6 | 193 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9000    0.8086    0.8519       256
     Malignant_lesions     0.8341    0.8619    0.8478       210
body_of_stomach_normal     0.8616    0.9415    0.8998       205

              accuracy                         0.8659       671
             macro avg     0.8652    0.8707    0.8665       671
          weighted avg     0.8676    0.8659    0.8652       671

```
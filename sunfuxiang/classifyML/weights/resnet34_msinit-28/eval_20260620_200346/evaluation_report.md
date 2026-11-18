# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-28/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8968 |
| Accuracy | 0.8957 |
| Macro-AUROC (OvR) | 0.9772 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8948 |
| Balanced Acc | 0.9013 |
| Macro-Sensitivity | 0.9013 |
| Macro-Specificity | 0.9482 |
| Macro-Precision | 0.8953 |
| AUC-PR (Macro) | 0.9590 |
| Cohen's Kappa | 0.8434 |
| MCC | 0.8449 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 1001.8 (1.00 ms/张) |
| End-to-End FPS | 215.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8281 | 0.9590 | 0.9258 | 0.8742 | 0.9005 | 0.9674 | 0.9559 | 256 |
| Malignant_lesions | 0.8905 | 0.9262 | 0.8462 | 0.8677 | 0.9489 | 0.9674 | 0.9288 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9592 | 0.9140 | 0.9484 | 0.9933 | 0.9967 | 0.9924 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=212 FP=17 TN=398 FN=44
- **Malignant_lesions**: TP=187 FP=34 TN=427 FN=23
- **body_of_stomach_normal**: TP=202 FP=19 TN=447 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 212 | 32 | 12 |
| Malignant_lesions | 16 | 187 | 7 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9258    0.8281    0.8742       256
     Malignant_lesions     0.8462    0.8905    0.8677       210
body_of_stomach_normal     0.9140    0.9854    0.9484       205

              accuracy                         0.8957       671
             macro avg     0.8953    0.9013    0.8968       671
          weighted avg     0.8973    0.8957    0.8948       671

```
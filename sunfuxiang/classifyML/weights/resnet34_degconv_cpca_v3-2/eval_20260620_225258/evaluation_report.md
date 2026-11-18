# resnet34_degconv_cpca_v3 评估报告

- 权重: `./weights/resnet34_degconv_cpca_v3-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9050 |
| Accuracy | 0.9031 |
| Macro-AUROC (OvR) | 0.9809 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9027 |
| Balanced Acc | 0.9075 |
| Macro-Sensitivity | 0.9075 |
| Macro-Specificity | 0.9513 |
| Macro-Precision | 0.9034 |
| AUC-PR (Macro) | 0.9627 |
| Cohen's Kappa | 0.8543 |
| MCC | 0.8549 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.76G |
| FLOPs | 7.53G |
| Params | 21.86M |
| Pure-Inference FPS | 3515.9 (0.28 ms/张) |
| End-to-End FPS | 223.8 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9470 | 0.9083 | 0.8790 | 0.9118 | 0.9743 | 0.9636 | 256 |
| Malignant_lesions | 0.8905 | 0.9306 | 0.8539 | 0.8718 | 0.9491 | 0.9721 | 0.9354 | 210 |
| body_of_stomach_normal | 0.9805 | 0.9764 | 0.9481 | 0.9640 | 0.9913 | 0.9964 | 0.9891 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=22 TN=393 FN=38
- **Malignant_lesions**: TP=187 FP=32 TN=429 FN=23
- **body_of_stomach_normal**: TP=201 FP=11 TN=455 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 31 | 7 |
| Malignant_lesions | 19 | 187 | 4 |
| body_of_stomach_normal | 3 | 1 | 201 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9083    0.8516    0.8790       256
     Malignant_lesions     0.8539    0.8905    0.8718       210
body_of_stomach_normal     0.9481    0.9805    0.9640       205

              accuracy                         0.9031       671
             macro avg     0.9034    0.9075    0.9050       671
          weighted avg     0.9034    0.9031    0.9027       671

```
# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-8/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8969 |
| Accuracy | 0.8957 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8949 |
| Balanced Acc | 0.9016 |
| Macro-Sensitivity | 0.9016 |
| Macro-Specificity | 0.9482 |
| Macro-Precision | 0.8957 |
| AUC-PR (Macro) | 0.9648 |
| Cohen's Kappa | 0.8434 |
| MCC | 0.8452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.71G |
| FLOPs | 9.42G |
| Params | 24.58M |
| Pure-Inference FPS | 3060.4 (0.33 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8242 | 0.9614 | 0.9295 | 0.8737 | 0.8986 | 0.9729 | 0.9633 | 256 |
| Malignant_lesions | 0.8952 | 0.9219 | 0.8393 | 0.8664 | 0.9508 | 0.9717 | 0.9375 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9614 | 0.9182 | 0.9506 | 0.9933 | 0.9972 | 0.9935 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=211 FP=16 TN=399 FN=45
- **Malignant_lesions**: TP=188 FP=36 TN=425 FN=22
- **body_of_stomach_normal**: TP=202 FP=18 TN=448 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 211 | 34 | 11 |
| Malignant_lesions | 15 | 188 | 7 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9295    0.8242    0.8737       256
     Malignant_lesions     0.8393    0.8952    0.8664       210
body_of_stomach_normal     0.9182    0.9854    0.9506       205

              accuracy                         0.8957       671
             macro avg     0.8957    0.9016    0.8969       671
          weighted avg     0.8978    0.8957    0.8949       671

```
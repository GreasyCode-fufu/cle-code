# resnet34_msinit 评估报告

- 权重: `./weights/resnet34_msinit-29/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9121 |
| Accuracy | 0.9106 |
| Macro-AUROC (OvR) | 0.9816 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9105 |
| Balanced Acc | 0.9130 |
| Macro-Sensitivity | 0.9130 |
| Macro-Specificity | 0.9548 |
| Macro-Precision | 0.9113 |
| AUC-PR (Macro) | 0.9677 |
| Cohen's Kappa | 0.8653 |
| MCC | 0.8654 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.78G |
| FLOPs | 7.57G |
| Params | 21.37M |
| Pure-Inference FPS | 999.4 (1.00 ms/张) |
| End-to-End FPS | 196.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8828 | 0.9422 | 0.9040 | 0.8933 | 0.9287 | 0.9730 | 0.9633 | 256 |
| Malignant_lesions | 0.9000 | 0.9458 | 0.8832 | 0.8915 | 0.9540 | 0.9759 | 0.9481 | 210 |
| body_of_stomach_normal | 0.9561 | 0.9764 | 0.9469 | 0.9515 | 0.9806 | 0.9960 | 0.9917 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=226 FP=24 TN=391 FN=30
- **Malignant_lesions**: TP=189 FP=25 TN=436 FN=21
- **body_of_stomach_normal**: TP=196 FP=11 TN=455 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 226 | 22 | 8 |
| Malignant_lesions | 18 | 189 | 3 |
| body_of_stomach_normal | 6 | 3 | 196 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9040    0.8828    0.8933       256
     Malignant_lesions     0.8832    0.9000    0.8915       210
body_of_stomach_normal     0.9469    0.9561    0.9515       205

              accuracy                         0.9106       671
             macro avg     0.9113    0.9130    0.9121       671
          weighted avg     0.9106    0.9106    0.9105       671

```
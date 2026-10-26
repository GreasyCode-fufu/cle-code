# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9439 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9915 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9443 |
| Balanced Acc | 0.9424 |
| Macro-Sensitivity | 0.9424 |
| Macro-Specificity | 0.9717 |
| Macro-Precision | 0.9468 |
| AUC-PR (Macro) | 0.9841 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9170 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1081.6 (0.92 ms/张) |
| End-to-End FPS | 232.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9429 | 0.9050 | 0.9324 | 0.9775 | 0.9878 | 0.9782 | 208 |
| Malignant_lesions | 0.8864 | 0.9800 | 0.9512 | 0.9176 | 0.9515 | 0.9871 | 0.9748 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=21 TN=347 FN=8
- **Malignant_lesions**: TP=156 FP=8 TN=392 FN=20
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 18 | 156 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9050    0.9615    0.9324       208
     Malignant_lesions     0.9512    0.8864    0.9176       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9444       576
             macro avg     0.9468    0.9424    0.9439       576
          weighted avg     0.9455    0.9444    0.9443       576

```
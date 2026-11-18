# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet34_seed42/resnet34/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9740 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9788 |
| Balanced Acc | 0.9691 |
| Macro-Sensitivity | 0.9691 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9794 |
| AUC-PR (Macro) | 0.9911 |
| Cohen's Kappa | 0.9663 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1099.4 (0.91 ms/张) |
| End-to-End FPS | 232.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9698 | 0.9650 | 0.9772 | 0.9912 | 0.9962 | 0.9949 | 195 |
| Malignant_lesions | 0.9241 | 0.9943 | 0.9733 | 0.9481 | 0.9830 | 0.9932 | 0.9785 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=7 TN=225 FN=2
- **Malignant_lesions**: TP=73 FP=2 TN=346 FN=6
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9650    0.9897    0.9772       195
     Malignant_lesions     0.9733    0.9241    0.9481        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9789       427
             macro avg     0.9794    0.9691    0.9740       427
          weighted avg     0.9791    0.9789    0.9788       427

```
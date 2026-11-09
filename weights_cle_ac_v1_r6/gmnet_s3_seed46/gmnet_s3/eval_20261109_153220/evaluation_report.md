# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed46/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9686 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9744 |
| Balanced Acc | 0.9737 |
| Macro-Sensitivity | 0.9737 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9641 |
| AUC-PR (Macro) | 0.9914 |
| Cohen's Kappa | 0.9592 |
| MCC | 0.9595 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1389.8 (0.72 ms/张) |
| End-to-End FPS | 205.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9914 | 0.9894 | 0.9740 | 0.9664 | 0.9978 | 0.9974 | 195 |
| Malignant_lesions | 0.9620 | 0.9799 | 0.9157 | 0.9383 | 0.9913 | 0.9949 | 0.9767 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=2 TN=230 FN=8
- **Malignant_lesions**: TP=76 FP=7 TN=341 FN=3
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 7 | 1 |
| Malignant_lesions | 2 | 76 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9894    0.9590    0.9740       195
     Malignant_lesions     0.9157    0.9620    0.9383        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9742       427
             macro avg     0.9641    0.9737    0.9686       427
          weighted avg     0.9749    0.9742    0.9744       427

```
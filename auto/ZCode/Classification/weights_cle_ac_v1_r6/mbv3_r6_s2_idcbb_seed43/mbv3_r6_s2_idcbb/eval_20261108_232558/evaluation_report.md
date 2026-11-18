# mbv3_r6_s2_idcbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_idcbb_seed43/mbv3_r6_s2_idcbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9438 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9442 |
| Balanced Acc | 0.9436 |
| Macro-Sensitivity | 0.9436 |
| Macro-Specificity | 0.9720 |
| Macro-Precision | 0.9444 |
| AUC-PR (Macro) | 0.9856 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9166 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.73M |
| FLOPs | 469.46M |
| Params | 3.26M |
| Pure-Inference FPS | 3725.5 (0.27 ms/张) |
| End-to-End FPS | 222.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9592 | 0.9282 | 0.9305 | 0.9619 | 0.9873 | 0.9798 | 208 |
| Malignant_lesions | 0.9034 | 0.9725 | 0.9353 | 0.9191 | 0.9581 | 0.9908 | 0.9783 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=15 TN=353 FN=14
- **Malignant_lesions**: TP=159 FP=11 TN=389 FN=17
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9282    0.9327    0.9305       208
     Malignant_lesions     0.9353    0.9034    0.9191       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9444       576
             macro avg     0.9444    0.9436    0.9438       576
          weighted avg     0.9442    0.9444    0.9442       576

```
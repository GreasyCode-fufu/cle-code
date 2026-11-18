# convnext_dph_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fgc_seed42/convnext_dph_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9559 |
| Macro-Sensitivity | 0.9559 |
| Macro-Specificity | 0.9778 |
| Macro-Precision | 0.9540 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9327 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 28.59M |
| Pure-Inference FPS | 640.7 (1.56 ms/张) |
| End-to-End FPS | 205.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9864 | 0.9745 | 0.9455 | 0.9553 | 0.9961 | 0.9934 | 208 |
| Malignant_lesions | 0.9545 | 0.9600 | 0.9130 | 0.9333 | 0.9796 | 0.9927 | 0.9852 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=5 TN=363 FN=17
- **Malignant_lesions**: TP=168 FP=16 TN=384 FN=8
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 16 | 1 |
| Malignant_lesions | 4 | 168 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9745    0.9183    0.9455       208
     Malignant_lesions     0.9130    0.9545    0.9333       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9549       576
             macro avg     0.9540    0.9559    0.9545       576
          weighted avg     0.9557    0.9549    0.9548       576

```
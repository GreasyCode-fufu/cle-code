# fast_t0_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_seed44/fast_t0_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9353 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9929 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9356 |
| Balanced Acc | 0.9349 |
| Macro-Sensitivity | 0.9349 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.9359 |
| AUC-PR (Macro) | 0.9858 |
| Cohen's Kappa | 0.9034 |
| MCC | 0.9035 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 4585.1 (0.22 ms/张) |
| End-to-End FPS | 124.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9538 | 0.9190 | 0.9234 | 0.9590 | 0.9896 | 0.9843 | 208 |
| Malignant_lesions | 0.8977 | 0.9700 | 0.9294 | 0.9133 | 0.9557 | 0.9898 | 0.9743 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9792 | 0.9592 | 0.9691 | 0.9895 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=17 TN=351 FN=15
- **Malignant_lesions**: TP=158 FP=12 TN=388 FN=18
- **body_of_stomach_normal**: TP=188 FP=8 TN=376 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 12 | 3 |
| Malignant_lesions | 13 | 158 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9190    0.9279    0.9234       208
     Malignant_lesions     0.9294    0.8977    0.9133       176
body_of_stomach_normal     0.9592    0.9792    0.9691       192

              accuracy                         0.9358       576
             macro avg     0.9359    0.9349    0.9353       576
          weighted avg     0.9356    0.9358    0.9356       576

```
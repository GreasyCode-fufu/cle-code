# convnext_v3cs 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cs_seed42/convnext_v3cs/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9583 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9586 |
| Balanced Acc | 0.9596 |
| Macro-Sensitivity | 0.9596 |
| Macro-Specificity | 0.9796 |
| Macro-Precision | 0.9585 |
| AUC-PR (Macro) | 0.9915 |
| Cohen's Kappa | 0.9375 |
| MCC | 0.9382 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.29G |
| Params | 28.40M |
| Pure-Inference FPS | 661.6 (1.51 ms/张) |
| End-to-End FPS | 203.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9864 | 0.9750 | 0.9559 | 0.9654 | 0.9935 | 0.9899 | 208 |
| Malignant_lesions | 0.9830 | 0.9550 | 0.9058 | 0.9428 | 0.9922 | 0.9935 | 0.9857 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9974 | 0.9946 | 0.9761 | 0.9795 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=5 TN=363 FN=13
- **Malignant_lesions**: TP=173 FP=18 TN=382 FN=3
- **body_of_stomach_normal**: TP=184 FP=1 TN=383 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 3 | 173 | 0 |
| body_of_stomach_normal | 2 | 6 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9750    0.9375    0.9559       208
     Malignant_lesions     0.9058    0.9830    0.9428       176
body_of_stomach_normal     0.9946    0.9583    0.9761       192

              accuracy                         0.9583       576
             macro avg     0.9585    0.9596    0.9583       576
          weighted avg     0.9604    0.9583    0.9586       576

```
# convnext_dph_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fgc_seed44/convnext_dph_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9532 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9533 |
| Balanced Acc | 0.9553 |
| Macro-Sensitivity | 0.9553 |
| Macro-Specificity | 0.9771 |
| Macro-Precision | 0.9536 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9297 |
| MCC | 0.9310 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 28.59M |
| Pure-Inference FPS | 693.2 (1.44 ms/张) |
| End-to-End FPS | 207.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9087 | 0.9891 | 0.9793 | 0.9426 | 0.9504 | 0.9929 | 0.9898 | 208 |
| Malignant_lesions | 0.9886 | 0.9475 | 0.8923 | 0.9380 | 0.9948 | 0.9945 | 0.9877 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=4 TN=364 FN=19
- **Malignant_lesions**: TP=174 FP=21 TN=379 FN=2
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 18 | 1 |
| Malignant_lesions | 1 | 174 | 1 |
| body_of_stomach_normal | 3 | 3 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9793    0.9087    0.9426       208
     Malignant_lesions     0.8923    0.9886    0.9380       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9531       576
             macro avg     0.9536    0.9553    0.9532       576
          weighted avg     0.9561    0.9531    0.9533       576

```
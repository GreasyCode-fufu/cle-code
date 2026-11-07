# convnext_v5se 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5se_seed42/convnext_v5se/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9531 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9533 |
| Balanced Acc | 0.9542 |
| Macro-Sensitivity | 0.9542 |
| Macro-Specificity | 0.9769 |
| Macro-Precision | 0.9529 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9296 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.47M |
| Pure-Inference FPS | 471.4 (2.12 ms/张) |
| End-to-End FPS | 185.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9783 | 0.9602 | 0.9438 | 0.9600 | 0.9946 | 0.9915 | 208 |
| Malignant_lesions | 0.9659 | 0.9575 | 0.9091 | 0.9366 | 0.9846 | 0.9955 | 0.9903 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=360 FN=15
- **Malignant_lesions**: TP=170 FP=17 TN=383 FN=6
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 3 | 3 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9279    0.9438       208
     Malignant_lesions     0.9091    0.9659    0.9366       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9531       576
             macro avg     0.9529    0.9542    0.9531       576
          weighted avg     0.9543    0.9531    0.9533       576

```
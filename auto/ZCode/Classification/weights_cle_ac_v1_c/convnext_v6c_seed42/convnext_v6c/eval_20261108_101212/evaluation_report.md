# convnext_v6c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6c_seed42/convnext_v6c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9666 |
| Macro-Sensitivity | 0.9666 |
| Macro-Specificity | 0.9829 |
| Macro-Precision | 0.9648 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9479 |
| MCC | 0.9483 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.78G |
| FLOPs | 11.57G |
| Params | 30.61M |
| Pure-Inference FPS | 472.0 (2.12 ms/张) |
| End-to-End FPS | 175.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9864 | 0.9747 | 0.9507 | 0.9603 | 0.9951 | 0.9924 | 208 |
| Malignant_lesions | 0.9773 | 0.9650 | 0.9247 | 0.9503 | 0.9897 | 0.9962 | 0.9908 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9974 | 0.9948 | 0.9948 | 0.9974 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=5 TN=363 FN=15
- **Malignant_lesions**: TP=172 FP=14 TN=386 FN=4
- **body_of_stomach_normal**: TP=191 FP=1 TN=383 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 4 | 172 | 0 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9747    0.9279    0.9507       208
     Malignant_lesions     0.9247    0.9773    0.9503       176
body_of_stomach_normal     0.9948    0.9948    0.9948       192

              accuracy                         0.9653       576
             macro avg     0.9648    0.9666    0.9653       576
          weighted avg     0.9661    0.9653    0.9653       576

```
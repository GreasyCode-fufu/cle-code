# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/jumbo_small_seed46/jumbo_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9255 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9258 |
| Balanced Acc | 0.9235 |
| Macro-Sensitivity | 0.9235 |
| Macro-Specificity | 0.9621 |
| Macro-Precision | 0.9291 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.8876 |
| MCC | 0.8885 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 779.1 (1.28 ms/张) |
| End-to-End FPS | 193.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9239 | 0.8756 | 0.9099 | 0.9687 | 0.9876 | 0.9798 | 208 |
| Malignant_lesions | 0.8807 | 0.9650 | 0.9172 | 0.8986 | 0.9484 | 0.9862 | 0.9718 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9974 | 0.9945 | 0.9679 | 0.9721 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=28 TN=340 FN=11
- **Malignant_lesions**: TP=155 FP=14 TN=386 FN=21
- **body_of_stomach_normal**: TP=181 FP=1 TN=383 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 21 | 155 | 0 |
| body_of_stomach_normal | 7 | 4 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8756    0.9471    0.9099       208
     Malignant_lesions     0.9172    0.8807    0.8986       176
body_of_stomach_normal     0.9945    0.9427    0.9679       192

              accuracy                         0.9253       576
             macro avg     0.9291    0.9235    0.9255       576
          weighted avg     0.9279    0.9253    0.9258       576

```
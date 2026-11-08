# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed44/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9404 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9407 |
| Balanced Acc | 0.9379 |
| Macro-Sensitivity | 0.9379 |
| Macro-Specificity | 0.9696 |
| Macro-Precision | 0.9471 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9128 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5337.0 (0.19 ms/张) |
| End-to-End FPS | 119.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9321 | 0.8904 | 0.9312 | 0.9856 | 0.9907 | 0.9844 | 208 |
| Malignant_lesions | 0.8636 | 0.9950 | 0.9870 | 0.9212 | 0.9431 | 0.9917 | 0.9838 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=25 TN=343 FN=5
- **Malignant_lesions**: TP=152 FP=2 TN=398 FN=24
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 2 | 3 |
| Malignant_lesions | 20 | 152 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8904    0.9760    0.9312       208
     Malignant_lesions     0.9870    0.8636    0.9212       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9410       576
             macro avg     0.9471    0.9379    0.9404       576
          weighted avg     0.9444    0.9410    0.9407       576

```
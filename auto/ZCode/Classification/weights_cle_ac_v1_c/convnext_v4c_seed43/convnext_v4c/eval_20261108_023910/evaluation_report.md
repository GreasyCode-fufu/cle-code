# convnext_v4c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4c_seed43/convnext_v4c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9349 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9929 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9344 |
| Balanced Acc | 0.9322 |
| Macro-Sensitivity | 0.9322 |
| Macro-Specificity | 0.9661 |
| Macro-Precision | 0.9400 |
| AUC-PR (Macro) | 0.9869 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9018 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.85G |
| FLOPs | 3.71G |
| Params | 3.45M |
| Pure-Inference FPS | 772.6 (1.29 ms/张) |
| End-to-End FPS | 223.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9239 | 0.8767 | 0.9149 | 0.9742 | 0.9883 | 0.9777 | 208 |
| Malignant_lesions | 0.8920 | 0.9900 | 0.9752 | 0.9318 | 0.9542 | 0.9923 | 0.9872 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9844 | 0.9681 | 0.9579 | 0.9742 | 0.9980 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=28 TN=340 FN=9
- **Malignant_lesions**: TP=157 FP=4 TN=396 FN=19
- **body_of_stomach_normal**: TP=182 FP=6 TN=378 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 4 | 5 |
| Malignant_lesions | 18 | 157 | 1 |
| body_of_stomach_normal | 10 | 0 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8767    0.9567    0.9149       208
     Malignant_lesions     0.9752    0.8920    0.9318       176
body_of_stomach_normal     0.9681    0.9479    0.9579       192

              accuracy                         0.9340       576
             macro avg     0.9400    0.9322    0.9349       576
          weighted avg     0.9372    0.9340    0.9344       576

```
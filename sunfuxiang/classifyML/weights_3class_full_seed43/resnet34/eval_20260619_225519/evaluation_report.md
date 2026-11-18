# resnet34 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed43/resnet34/resnet34_3class_full_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8773 |
| Accuracy | 0.8763 |
| Macro-AUROC (OvR) | 0.9650 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8754 |
| Balanced Acc | 0.8818 |
| Macro-Sensitivity | 0.8818 |
| Macro-Specificity | 0.9385 |
| Macro-Precision | 0.8758 |
| AUC-PR (Macro) | 0.9325 |
| Cohen's Kappa | 0.8143 |
| MCC | 0.8158 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3791.6 (0.26 ms/张) |
| End-to-End FPS | 232.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8125 | 0.9494 | 0.9083 | 0.8577 | 0.8914 | 0.9520 | 0.9437 | 256 |
| Malignant_lesions | 0.8476 | 0.9089 | 0.8091 | 0.8279 | 0.9290 | 0.9464 | 0.8624 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9571 | 0.9099 | 0.9461 | 0.9933 | 0.9965 | 0.9913 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=208 FP=21 TN=394 FN=48
- **Malignant_lesions**: TP=178 FP=42 TN=419 FN=32
- **body_of_stomach_normal**: TP=202 FP=20 TN=446 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 208 | 40 | 8 |
| Malignant_lesions | 20 | 178 | 12 |
| body_of_stomach_normal | 1 | 2 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9083    0.8125    0.8577       256
     Malignant_lesions     0.8091    0.8476    0.8279       210
body_of_stomach_normal     0.9099    0.9854    0.9461       205

              accuracy                         0.8763       671
             macro avg     0.8758    0.8818    0.8773       671
          weighted avg     0.8777    0.8763    0.8754       671

```
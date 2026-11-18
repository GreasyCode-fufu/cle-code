# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed44/vit_base_patch16_224/vit_base_patch16_224_3class_hq_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8853 |
| Accuracy | 0.8838 |
| Macro-AUROC (OvR) | 0.9723 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8841 |
| Balanced Acc | 0.8835 |
| Macro-Sensitivity | 0.8835 |
| Macro-Specificity | 0.9406 |
| Macro-Precision | 0.8877 |
| AUC-PR (Macro) | 0.9517 |
| Cohen's Kappa | 0.8244 |
| MCC | 0.8247 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 869.5 (1.15 ms/张) |
| End-to-End FPS | 201.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8906 | 0.9060 | 0.8539 | 0.8719 | 0.9307 | 0.9662 | 0.9480 | 256 |
| Malignant_lesions | 0.8476 | 0.9371 | 0.8599 | 0.8537 | 0.9310 | 0.9633 | 0.9265 | 210 |
| body_of_stomach_normal | 0.9122 | 0.9785 | 0.9492 | 0.9303 | 0.9620 | 0.9875 | 0.9807 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=228 FP=39 TN=376 FN=28
- **Malignant_lesions**: TP=178 FP=29 TN=432 FN=32
- **body_of_stomach_normal**: TP=187 FP=10 TN=456 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 228 | 20 | 8 |
| Malignant_lesions | 30 | 178 | 2 |
| body_of_stomach_normal | 9 | 9 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8539    0.8906    0.8719       256
     Malignant_lesions     0.8599    0.8476    0.8537       210
body_of_stomach_normal     0.9492    0.9122    0.9303       205

              accuracy                         0.8838       671
             macro avg     0.8877    0.8835    0.8853       671
          weighted avg     0.8849    0.8838    0.8841       671

```
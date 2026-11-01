# convnext_fgc34_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc34_dsd2_seed42/convnext_fgc34_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9113 |
| Accuracy | 0.9132 |
| Macro-AUROC (OvR) | 0.9848 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9121 |
| Balanced Acc | 0.9081 |
| Macro-Sensitivity | 0.9081 |
| Macro-Specificity | 0.9553 |
| Macro-Precision | 0.9241 |
| AUC-PR (Macro) | 0.9730 |
| Cohen's Kappa | 0.8689 |
| MCC | 0.8732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.40G |
| FLOPs | 8.80G |
| Params | 27.49M |
| Pure-Inference FPS | 664.5 (1.50 ms/张) |
| End-to-End FPS | 201.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.8913 | 0.8333 | 0.8929 | 0.9762 | 0.9790 | 0.9644 | 208 |
| Malignant_lesions | 0.7784 | 0.9875 | 0.9648 | 0.8616 | 0.9101 | 0.9780 | 0.9580 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9975 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=40 TN=328 FN=8
- **Malignant_lesions**: TP=137 FP=5 TN=395 FN=39
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 37 | 137 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8333    0.9615    0.8929       208
     Malignant_lesions     0.9648    0.7784    0.8616       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9132       576
             macro avg     0.9241    0.9081    0.9113       576
          weighted avg     0.9205    0.9132    0.9121       576

```
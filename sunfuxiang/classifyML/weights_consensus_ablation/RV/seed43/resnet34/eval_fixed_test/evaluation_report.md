# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RV/seed43/resnet34/resnet34_RV_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9064 |
| Accuracy | 0.9046 |
| Macro-AUROC (OvR) | 0.9801 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9043 |
| Balanced Acc | 0.9107 |
| Macro-Sensitivity | 0.9107 |
| Macro-Specificity | 0.9528 |
| Macro-Precision | 0.9065 |
| AUC-PR (Macro) | 0.9640 |
| Cohen's Kappa | 0.8569 |
| MCC | 0.8592 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3776.8 (0.26 ms/张) |
| End-to-End FPS | 238.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8281 | 0.9687 | 0.9422 | 0.8815 | 0.9013 | 0.9732 | 0.9639 | 256 |
| Malignant_lesions | 0.9333 | 0.9154 | 0.8340 | 0.8809 | 0.9679 | 0.9710 | 0.9377 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9742 | 0.9431 | 0.9567 | 0.9870 | 0.9960 | 0.9903 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=212 FP=13 TN=402 FN=44
- **Malignant_lesions**: TP=196 FP=39 TN=422 FN=14
- **body_of_stomach_normal**: TP=199 FP=12 TN=454 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 212 | 34 | 10 |
| Malignant_lesions | 12 | 196 | 2 |
| body_of_stomach_normal | 1 | 5 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9422    0.8281    0.8815       256
     Malignant_lesions     0.8340    0.9333    0.8809       210
body_of_stomach_normal     0.9431    0.9707    0.9567       205

              accuracy                         0.9046       671
             macro avg     0.9065    0.9107    0.9064       671
          weighted avg     0.9086    0.9046    0.9043       671

```
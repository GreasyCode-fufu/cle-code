# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RC/seed43/resnet34/resnet34_RC_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9225 |
| Accuracy | 0.9210 |
| Macro-AUROC (OvR) | 0.9777 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9210 |
| Balanced Acc | 0.9248 |
| Macro-Sensitivity | 0.9248 |
| Macro-Specificity | 0.9605 |
| Macro-Precision | 0.9216 |
| AUC-PR (Macro) | 0.9569 |
| Cohen's Kappa | 0.8812 |
| MCC | 0.8819 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3795.1 (0.26 ms/张) |
| End-to-End FPS | 243.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8750 | 0.9639 | 0.9372 | 0.9051 | 0.9259 | 0.9667 | 0.9624 | 256 |
| Malignant_lesions | 0.9238 | 0.9349 | 0.8661 | 0.8940 | 0.9642 | 0.9700 | 0.9176 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9828 | 0.9615 | 0.9685 | 0.9892 | 0.9963 | 0.9908 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=224 FP=15 TN=400 FN=32
- **Malignant_lesions**: TP=194 FP=30 TN=431 FN=16
- **body_of_stomach_normal**: TP=200 FP=8 TN=458 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 27 | 5 |
| Malignant_lesions | 13 | 194 | 3 |
| body_of_stomach_normal | 2 | 3 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9372    0.8750    0.9051       256
     Malignant_lesions     0.8661    0.9238    0.8940       210
body_of_stomach_normal     0.9615    0.9756    0.9685       205

              accuracy                         0.9210       671
             macro avg     0.9216    0.9248    0.9225       671
          weighted avg     0.9224    0.9210    0.9210       671

```
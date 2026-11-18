# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/VC/seed43/resnet34/resnet34_VC_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9020 |
| Accuracy | 0.9001 |
| Macro-AUROC (OvR) | 0.9779 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9003 |
| Balanced Acc | 0.9065 |
| Macro-Sensitivity | 0.9065 |
| Macro-Specificity | 0.9508 |
| Macro-Precision | 0.9039 |
| AUC-PR (Macro) | 0.9595 |
| Cohen's Kappa | 0.8503 |
| MCC | 0.8536 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3761.5 (0.27 ms/张) |
| End-to-End FPS | 248.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8203 | 0.9759 | 0.9545 | 0.8824 | 0.8980 | 0.9695 | 0.9598 | 256 |
| Malignant_lesions | 0.9333 | 0.9002 | 0.8099 | 0.8673 | 0.9674 | 0.9676 | 0.9268 | 210 |
| body_of_stomach_normal | 0.9659 | 0.9764 | 0.9474 | 0.9565 | 0.9848 | 0.9966 | 0.9920 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=210 FP=10 TN=405 FN=46
- **Malignant_lesions**: TP=196 FP=46 TN=415 FN=14
- **body_of_stomach_normal**: TP=198 FP=11 TN=455 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 210 | 39 | 7 |
| Malignant_lesions | 10 | 196 | 4 |
| body_of_stomach_normal | 0 | 7 | 198 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9545    0.8203    0.8824       256
     Malignant_lesions     0.8099    0.9333    0.8673       210
body_of_stomach_normal     0.9474    0.9659    0.9565       205

              accuracy                         0.9001       671
             macro avg     0.9039    0.9065    0.9020       671
          weighted avg     0.9071    0.9001    0.9003       671

```
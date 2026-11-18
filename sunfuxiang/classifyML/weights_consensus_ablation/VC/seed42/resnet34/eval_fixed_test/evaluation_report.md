# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/VC/seed42/resnet34/resnet34_VC_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8846 |
| Accuracy | 0.8838 |
| Macro-AUROC (OvR) | 0.9800 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8824 |
| Balanced Acc | 0.8895 |
| Macro-Sensitivity | 0.8895 |
| Macro-Specificity | 0.9421 |
| Macro-Precision | 0.8833 |
| AUC-PR (Macro) | 0.9649 |
| Cohen's Kappa | 0.8255 |
| MCC | 0.8273 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3798.5 (0.26 ms/张) |
| End-to-End FPS | 235.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8164 | 0.9470 | 0.9048 | 0.8583 | 0.8932 | 0.9715 | 0.9599 | 256 |
| Malignant_lesions | 0.8667 | 0.9414 | 0.8708 | 0.8687 | 0.9394 | 0.9719 | 0.9413 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9378 | 0.8745 | 0.9266 | 0.9932 | 0.9968 | 0.9933 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=209 FP=22 TN=393 FN=47
- **Malignant_lesions**: TP=182 FP=27 TN=434 FN=28
- **body_of_stomach_normal**: TP=202 FP=29 TN=437 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 209 | 26 | 21 |
| Malignant_lesions | 20 | 182 | 8 |
| body_of_stomach_normal | 2 | 1 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9048    0.8164    0.8583       256
     Malignant_lesions     0.8708    0.8667    0.8687       210
body_of_stomach_normal     0.8745    0.9854    0.9266       205

              accuracy                         0.8838       671
             macro avg     0.8833    0.8895    0.8846       671
          weighted avg     0.8849    0.8838    0.8824       671

```
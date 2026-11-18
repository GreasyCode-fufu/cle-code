# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RC/seed44/resnet34/resnet34_RC_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9147 |
| Accuracy | 0.9136 |
| Macro-AUROC (OvR) | 0.9801 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9131 |
| Balanced Acc | 0.9170 |
| Macro-Sensitivity | 0.9170 |
| Macro-Specificity | 0.9566 |
| Macro-Precision | 0.9132 |
| AUC-PR (Macro) | 0.9643 |
| Cohen's Kappa | 0.8699 |
| MCC | 0.8703 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3716.7 (0.27 ms/张) |
| End-to-End FPS | 231.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8750 | 0.9518 | 0.9180 | 0.8960 | 0.9251 | 0.9713 | 0.9616 | 256 |
| Malignant_lesions | 0.8857 | 0.9436 | 0.8774 | 0.8815 | 0.9477 | 0.9727 | 0.9402 | 210 |
| body_of_stomach_normal | 0.9902 | 0.9742 | 0.9442 | 0.9667 | 0.9956 | 0.9964 | 0.9913 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=224 FP=20 TN=395 FN=32
- **Malignant_lesions**: TP=186 FP=26 TN=435 FN=24
- **body_of_stomach_normal**: TP=203 FP=12 TN=454 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 224 | 25 | 7 |
| Malignant_lesions | 19 | 186 | 5 |
| body_of_stomach_normal | 1 | 1 | 203 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9180    0.8750    0.8960       256
     Malignant_lesions     0.8774    0.8857    0.8815       210
body_of_stomach_normal     0.9442    0.9902    0.9667       205

              accuracy                         0.9136       671
             macro avg     0.9132    0.9170    0.9147       671
          weighted avg     0.9133    0.9136    0.9131       671

```
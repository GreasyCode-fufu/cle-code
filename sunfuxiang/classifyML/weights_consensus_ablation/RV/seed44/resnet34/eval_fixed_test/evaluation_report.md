# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RV/seed44/resnet34/resnet34_RV_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9101 |
| Accuracy | 0.9091 |
| Macro-AUROC (OvR) | 0.9815 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9087 |
| Balanced Acc | 0.9127 |
| Macro-Sensitivity | 0.9127 |
| Macro-Specificity | 0.9545 |
| Macro-Precision | 0.9086 |
| AUC-PR (Macro) | 0.9674 |
| Cohen's Kappa | 0.8633 |
| MCC | 0.8638 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3783.8 (0.26 ms/张) |
| End-to-End FPS | 226.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8672 | 0.9542 | 0.9212 | 0.8934 | 0.9209 | 0.9748 | 0.9655 | 256 |
| Malignant_lesions | 0.8952 | 0.9414 | 0.8744 | 0.8847 | 0.9518 | 0.9739 | 0.9465 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9678 | 0.9302 | 0.9524 | 0.9890 | 0.9958 | 0.9903 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=222 FP=19 TN=396 FN=34
- **Malignant_lesions**: TP=188 FP=27 TN=434 FN=22
- **body_of_stomach_normal**: TP=200 FP=15 TN=451 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 222 | 24 | 10 |
| Malignant_lesions | 17 | 188 | 5 |
| body_of_stomach_normal | 2 | 3 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9212    0.8672    0.8934       256
     Malignant_lesions     0.8744    0.8952    0.8847       210
body_of_stomach_normal     0.9302    0.9756    0.9524       205

              accuracy                         0.9091       671
             macro avg     0.9086    0.9127    0.9101       671
          weighted avg     0.9093    0.9091    0.9087       671

```
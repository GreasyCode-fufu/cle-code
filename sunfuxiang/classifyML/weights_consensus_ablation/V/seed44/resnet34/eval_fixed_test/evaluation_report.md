# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/V/seed44/resnet34/resnet34_V_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9072 |
| Accuracy | 0.9061 |
| Macro-AUROC (OvR) | 0.9792 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9058 |
| Balanced Acc | 0.9080 |
| Macro-Sensitivity | 0.9080 |
| Macro-Specificity | 0.9525 |
| Macro-Precision | 0.9064 |
| AUC-PR (Macro) | 0.9628 |
| Cohen's Kappa | 0.8585 |
| MCC | 0.8586 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3786.9 (0.26 ms/张) |
| End-to-End FPS | 235.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8867 | 0.9398 | 0.9008 | 0.8937 | 0.9308 | 0.9727 | 0.9611 | 256 |
| Malignant_lesions | 0.8667 | 0.9414 | 0.8708 | 0.8687 | 0.9394 | 0.9680 | 0.9347 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9764 | 0.9476 | 0.9590 | 0.9870 | 0.9969 | 0.9924 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=227 FP=25 TN=390 FN=29
- **Malignant_lesions**: TP=182 FP=27 TN=434 FN=28
- **body_of_stomach_normal**: TP=199 FP=11 TN=455 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 227 | 22 | 7 |
| Malignant_lesions | 24 | 182 | 4 |
| body_of_stomach_normal | 1 | 5 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9008    0.8867    0.8937       256
     Malignant_lesions     0.8708    0.8667    0.8687       210
body_of_stomach_normal     0.9476    0.9707    0.9590       205

              accuracy                         0.9061       671
             macro avg     0.9064    0.9080    0.9072       671
          weighted avg     0.9057    0.9061    0.9058       671

```
# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/C/seed43/resnet34/resnet34_C_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8912 |
| Accuracy | 0.8897 |
| Macro-AUROC (OvR) | 0.9748 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8891 |
| Balanced Acc | 0.8946 |
| Macro-Sensitivity | 0.8946 |
| Macro-Specificity | 0.9448 |
| Macro-Precision | 0.8895 |
| AUC-PR (Macro) | 0.9534 |
| Cohen's Kappa | 0.8342 |
| MCC | 0.8351 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3751.3 (0.27 ms/张) |
| End-to-End FPS | 229.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8320 | 0.9446 | 0.9025 | 0.8659 | 0.9011 | 0.9642 | 0.9523 | 256 |
| Malignant_lesions | 0.8810 | 0.9262 | 0.8447 | 0.8625 | 0.9447 | 0.9650 | 0.9202 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9635 | 0.9213 | 0.9454 | 0.9868 | 0.9952 | 0.9877 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=213 FP=23 TN=392 FN=43
- **Malignant_lesions**: TP=185 FP=34 TN=427 FN=25
- **body_of_stomach_normal**: TP=199 FP=17 TN=449 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 213 | 30 | 13 |
| Malignant_lesions | 21 | 185 | 4 |
| body_of_stomach_normal | 2 | 4 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9025    0.8320    0.8659       256
     Malignant_lesions     0.8447    0.8810    0.8625       210
body_of_stomach_normal     0.9213    0.9707    0.9454       205

              accuracy                         0.8897       671
             macro avg     0.8895    0.8946    0.8912       671
          weighted avg     0.8902    0.8897    0.8891       671

```
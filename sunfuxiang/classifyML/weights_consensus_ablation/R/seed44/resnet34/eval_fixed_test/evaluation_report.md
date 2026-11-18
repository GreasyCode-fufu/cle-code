# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/R/seed44/resnet34/resnet34_R_seed44_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8938 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9757 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8923 |
| Balanced Acc | 0.8928 |
| Macro-Sensitivity | 0.8928 |
| Macro-Specificity | 0.9450 |
| Macro-Precision | 0.8958 |
| AUC-PR (Macro) | 0.9565 |
| Cohen's Kappa | 0.8379 |
| MCC | 0.8384 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3773.2 (0.27 ms/张) |
| End-to-End FPS | 238.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8984 | 0.9084 | 0.8582 | 0.8779 | 0.9355 | 0.9656 | 0.9541 | 256 |
| Malignant_lesions | 0.8190 | 0.9479 | 0.8776 | 0.8473 | 0.9200 | 0.9659 | 0.9259 | 210 |
| body_of_stomach_normal | 0.9610 | 0.9785 | 0.9517 | 0.9563 | 0.9828 | 0.9956 | 0.9896 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=230 FP=38 TN=377 FN=26
- **Malignant_lesions**: TP=172 FP=24 TN=437 FN=38
- **body_of_stomach_normal**: TP=197 FP=10 TN=456 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 230 | 19 | 7 |
| Malignant_lesions | 35 | 172 | 3 |
| body_of_stomach_normal | 3 | 5 | 197 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8582    0.8984    0.8779       256
     Malignant_lesions     0.8776    0.8190    0.8473       210
body_of_stomach_normal     0.9517    0.9610    0.9563       205

              accuracy                         0.8927       671
             macro avg     0.8958    0.8928    0.8938       671
          weighted avg     0.8928    0.8927    0.8923       671

```
# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/V/seed42/resnet34/resnet34_V_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9112 |
| Accuracy | 0.9106 |
| Macro-AUROC (OvR) | 0.9823 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9103 |
| Balanced Acc | 0.9134 |
| Macro-Sensitivity | 0.9134 |
| Macro-Specificity | 0.9554 |
| Macro-Precision | 0.9098 |
| AUC-PR (Macro) | 0.9689 |
| Cohen's Kappa | 0.8655 |
| MCC | 0.8659 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3779.9 (0.26 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8789 | 0.9590 | 0.9298 | 0.9036 | 0.9277 | 0.9758 | 0.9671 | 256 |
| Malignant_lesions | 0.8857 | 0.9371 | 0.8651 | 0.8753 | 0.9474 | 0.9746 | 0.9478 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9700 | 0.9346 | 0.9547 | 0.9891 | 0.9966 | 0.9916 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=225 FP=17 TN=398 FN=31
- **Malignant_lesions**: TP=186 FP=29 TN=432 FN=24
- **body_of_stomach_normal**: TP=200 FP=14 TN=452 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 225 | 24 | 7 |
| Malignant_lesions | 17 | 186 | 7 |
| body_of_stomach_normal | 0 | 5 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9298    0.8789    0.9036       256
     Malignant_lesions     0.8651    0.8857    0.8753       210
body_of_stomach_normal     0.9346    0.9756    0.9547       205

              accuracy                         0.9106       671
             macro avg     0.9098    0.9134    0.9112       671
          weighted avg     0.9110    0.9106    0.9103       671

```
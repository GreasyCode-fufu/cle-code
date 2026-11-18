# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/R/seed42/resnet34/resnet34_R_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8793 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9758 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8777 |
| Balanced Acc | 0.8807 |
| Macro-Sensitivity | 0.8807 |
| Macro-Specificity | 0.9385 |
| Macro-Precision | 0.8809 |
| AUC-PR (Macro) | 0.9552 |
| Cohen's Kappa | 0.8179 |
| MCC | 0.8191 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3725.2 (0.27 ms/张) |
| End-to-End FPS | 220.1 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8711 | 0.9084 | 0.8544 | 0.8627 | 0.9195 | 0.9627 | 0.9499 | 256 |
| Malignant_lesions | 0.7857 | 0.9523 | 0.8824 | 0.8312 | 0.9070 | 0.9681 | 0.9240 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9549 | 0.9058 | 0.9439 | 0.9933 | 0.9966 | 0.9918 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=223 FP=38 TN=377 FN=33
- **Malignant_lesions**: TP=165 FP=22 TN=439 FN=45
- **body_of_stomach_normal**: TP=202 FP=21 TN=445 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 223 | 21 | 12 |
| Malignant_lesions | 36 | 165 | 9 |
| body_of_stomach_normal | 2 | 1 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8544    0.8711    0.8627       256
     Malignant_lesions     0.8824    0.7857    0.8312       210
body_of_stomach_normal     0.9058    0.9854    0.9439       205

              accuracy                         0.8793       671
             macro avg     0.8809    0.8807    0.8793       671
          weighted avg     0.8789    0.8793    0.8777       671

```
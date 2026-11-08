# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7872 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9366 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8093 |
| Balanced Acc | 0.8230 |
| Macro-Sensitivity | 0.8230 |
| Macro-Specificity | 0.9034 |
| Macro-Precision | 0.7714 |
| AUC-PR (Macro) | 0.8873 |
| Cohen's Kappa | 0.6829 |
| MCC | 0.6899 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1536.4 (0.65 ms/张) |
| End-to-End FPS | 172.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7879 | 0.9038 | 0.9123 | 0.8455 | 0.7705 | 0.9178 | 0.9347 | 66 |
| malignant | 0.7812 | 0.9186 | 0.7812 | 0.7812 | 0.9186 | 0.9302 | 0.8161 | 32 |
| normal | 0.9000 | 0.8878 | 0.6207 | 0.7347 | 0.9775 | 0.9617 | 0.9110 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=52 FP=5 TN=47 FN=14
- **malignant**: TP=25 FP=7 TN=79 FN=7
- **normal**: TP=18 FP=11 TN=87 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 52 | 7 | 7 |
| malignant | 3 | 25 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9123    0.7879    0.8455        66
   malignant     0.7812    0.7812    0.7812        32
      normal     0.6207    0.9000    0.7347        20

    accuracy                         0.8051       118
   macro avg     0.7714    0.8230    0.7872       118
weighted avg     0.8273    0.8051    0.8093       118

```
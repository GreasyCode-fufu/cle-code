# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/lsnet_s_seed44/lsnet_s/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9824 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9823 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 5408.2 (0.18 ms/张) |
| End-to-End FPS | 785.8 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9944 | 0.9878 | 0.9918 | 0.9981 | 0.9962 | 0.9970 | 244 |
| NORMAL | 0.9778 | 0.9922 | 0.9851 | 0.9814 | 0.9884 | 0.9976 | 0.9958 | 270 |
| PNEUMONIA | 0.9741 | 0.9864 | 0.9741 | 0.9741 | 0.9864 | 0.9953 | 0.9874 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=3 TN=537 FN=1
- **NORMAL**: TP=264 FP=4 TN=510 FN=6
- **PNEUMONIA**: TP=263 FP=7 TN=507 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 264 | 6 |
| PNEUMONIA | 3 | 4 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9878    0.9959    0.9918       244
      NORMAL     0.9851    0.9778    0.9814       270
   PNEUMONIA     0.9741    0.9741    0.9741       270

    accuracy                         0.9821       784
   macro avg     0.9823    0.9826    0.9824       784
weighted avg     0.9821    0.9821    0.9821       784

```
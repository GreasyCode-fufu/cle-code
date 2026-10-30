# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9830 |
| AUC-PR (Macro) | 0.9931 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9734 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9715.1 (0.10 ms/张) |
| End-to-End FPS | 859.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9974 | 0.9971 | 244 |
| NORMAL | 0.9963 | 0.9786 | 0.9607 | 0.9782 | 0.9980 | 0.9984 | 0.9970 | 270 |
| PNEUMONIA | 0.9556 | 0.9961 | 0.9923 | 0.9736 | 0.9771 | 0.9943 | 0.9854 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=269 FP=11 TN=503 FN=1
- **PNEUMONIA**: TP=258 FP=2 TN=512 FN=12

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 269 | 1 |
| PNEUMONIA | 1 | 11 | 258 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9607    0.9963    0.9782       270
   PNEUMONIA     0.9923    0.9556    0.9736       270

    accuracy                         0.9821       784
   macro avg     0.9830    0.9826    0.9826       784
weighted avg     0.9825    0.9821    0.9821       784

```
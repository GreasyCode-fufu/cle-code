# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9826 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9978.0 (0.10 ms/张) |
| End-to-End FPS | 844.1 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9965 | 0.9969 | 244 |
| NORMAL | 0.9852 | 0.9844 | 0.9708 | 0.9779 | 0.9922 | 0.9973 | 0.9947 | 270 |
| PNEUMONIA | 0.9667 | 0.9903 | 0.9812 | 0.9739 | 0.9826 | 0.9948 | 0.9867 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=8 TN=506 FN=4
- **PNEUMONIA**: TP=261 FP=5 TN=509 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 8 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9708    0.9852    0.9779       270
   PNEUMONIA     0.9812    0.9667    0.9739       270

    accuracy                         0.9821       784
   macro avg     0.9826    0.9826    0.9826       784
weighted avg     0.9822    0.9821    0.9821       784

```
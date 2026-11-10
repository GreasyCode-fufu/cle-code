# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed42/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9827 |
| AUC-PR (Macro) | 0.9950 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 928.6 (1.08 ms/张) |
| End-to-End FPS | 455.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9981 | 0.9959 | 0.9938 | 0.9963 | 0.9956 | 0.9943 | 244 |
| NORMAL | 0.9852 | 0.9844 | 0.9708 | 0.9779 | 0.9922 | 0.9989 | 0.9980 | 270 |
| PNEUMONIA | 0.9704 | 0.9903 | 0.9813 | 0.9758 | 0.9845 | 0.9962 | 0.9926 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=539 FN=2
- **NORMAL**: TP=266 FP=8 TN=506 FN=4
- **PNEUMONIA**: TP=262 FP=5 TN=509 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9708    0.9852    0.9779       270
   PNEUMONIA     0.9813    0.9704    0.9758       270

    accuracy                         0.9821       784
   macro avg     0.9827    0.9825    0.9825       784
weighted avg     0.9822    0.9821    0.9821       784

```
# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed44/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9786 |
| Accuracy | 0.9783 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9783 |
| Balanced Acc | 0.9785 |
| Macro-Sensitivity | 0.9785 |
| Macro-Specificity | 0.9890 |
| Macro-Precision | 0.9789 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9674 |
| MCC | 0.9675 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 846.5 (1.18 ms/张) |
| End-to-End FPS | 441.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9836 | 0.9963 | 0.9917 | 0.9877 | 0.9926 | 0.9975 | 0.9961 | 244 |
| NORMAL | 0.9852 | 0.9805 | 0.9638 | 0.9744 | 0.9921 | 0.9975 | 0.9952 | 270 |
| PNEUMONIA | 0.9667 | 0.9903 | 0.9812 | 0.9739 | 0.9826 | 0.9951 | 0.9904 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=240 FP=2 TN=538 FN=4
- **NORMAL**: TP=266 FP=10 TN=504 FN=4
- **PNEUMONIA**: TP=261 FP=5 TN=509 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 240 | 3 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 2 | 7 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9917    0.9836    0.9877       244
      NORMAL     0.9638    0.9852    0.9744       270
   PNEUMONIA     0.9812    0.9667    0.9739       270

    accuracy                         0.9783       784
   macro avg     0.9789    0.9785    0.9786       784
weighted avg     0.9785    0.9783    0.9783       784

```
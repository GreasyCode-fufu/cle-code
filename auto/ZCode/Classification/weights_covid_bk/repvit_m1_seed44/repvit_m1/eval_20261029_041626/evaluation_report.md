# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9827 |
| AUC-PR (Macro) | 0.9967 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3897.9 (0.26 ms/张) |
| End-to-End FPS | 756.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9742 | 0.9922 | 0.9851 | 0.9796 | 0.9865 | 0.9975 | 0.9951 | 271 |
| PNEUMONIA | 0.9815 | 0.9825 | 0.9672 | 0.9743 | 0.9902 | 0.9971 | 0.9951 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=264 FP=4 TN=510 FN=7
- **PNEUMONIA**: TP=265 FP=9 TN=506 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9851    0.9742    0.9796       271
   PNEUMONIA     0.9672    0.9815    0.9743       270

    accuracy                         0.9822       785
   macro avg     0.9827    0.9825    0.9826       785
weighted avg     0.9823    0.9822    0.9822       785

```
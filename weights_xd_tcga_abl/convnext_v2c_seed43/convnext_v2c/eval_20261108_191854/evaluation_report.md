# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8074 |
| Accuracy | 0.8471 |
| Macro-AUROC (OvR) | 0.9624 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8444 |
| Balanced Acc | 0.8158 |
| Macro-Sensitivity | 0.8158 |
| Macro-Specificity | 0.9685 |
| Macro-Precision | 0.8026 |
| AUC-PR (Macro) | 0.8768 |
| Cohen's Kappa | 0.8034 |
| MCC | 0.8039 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 606.9 (1.65 ms/张) |
| End-to-End FPS | 317.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9938 | 0.7292 | 0.7973 | 0.9977 | 0.9968 | 0.9037 | 199 |
| cancer | 0.8820 | 0.9693 | 0.9050 | 0.8934 | 0.9613 | 0.9852 | 0.9649 | 2645 |
| empty | 0.9975 | 0.9604 | 0.9175 | 0.9559 | 0.9989 | 0.9823 | 0.9544 | 3256 |
| normal | 0.8196 | 0.9743 | 0.8613 | 0.8399 | 0.9652 | 0.9796 | 0.9248 | 1735 |
| other | 0.6395 | 0.9427 | 0.6824 | 0.6603 | 0.9314 | 0.8674 | 0.7314 | 1717 |
| stroma | 0.6765 | 0.9703 | 0.7205 | 0.6978 | 0.9636 | 0.9633 | 0.7819 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=65 TN=10370 FN=24
- **cancer**: TP=2333 FP=245 TN=7744 FN=312
- **empty**: TP=3248 FP=292 TN=7086 FN=8
- **normal**: TP=1422 FP=229 TN=8670 FN=313
- **other**: TP=1098 FP=511 TN=8406 FN=619
- **stroma**: TP=732 FP=284 TN=9268 FN=350

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 4 | 2 | 4 | 13 | 1 |
| cancer | 15 | 2333 | 2 | 103 | 143 | 49 |
| empty | 0 | 0 | 3248 | 0 | 8 | 0 |
| normal | 4 | 89 | 1 | 1422 | 152 | 67 |
| other | 43 | 107 | 287 | 15 | 1098 | 167 |
| stroma | 3 | 45 | 0 | 107 | 195 | 732 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7292    0.8794    0.7973       199
      cancer     0.9050    0.8820    0.8934      2645
       empty     0.9175    0.9975    0.9559      3256
      normal     0.8613    0.8196    0.8399      1735
       other     0.6824    0.6395    0.6603      1717
      stroma     0.7205    0.6765    0.6978      1082

    accuracy                         0.8471     10634
   macro avg     0.8026    0.8158    0.8074     10634
weighted avg     0.8437    0.8471    0.8444     10634

```
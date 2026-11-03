# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8050 |
| Accuracy | 0.8373 |
| Macro-AUROC (OvR) | 0.9751 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8386 |
| Balanced Acc | 0.8117 |
| Macro-Sensitivity | 0.8117 |
| Macro-Specificity | 0.9675 |
| Macro-Precision | 0.8194 |
| AUC-PR (Macro) | 0.8966 |
| Cohen's Kappa | 0.7947 |
| MCC | 0.7990 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 956.9 (1.04 ms/张) |
| End-to-End FPS | 389.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9952 | 0.7568 | 0.7962 | 0.9971 | 0.9965 | 0.9020 | 100 |
| cancer | 0.8051 | 0.9691 | 0.8880 | 0.8445 | 0.9424 | 0.9786 | 0.9321 | 1329 |
| empty | 0.9901 | 0.9985 | 0.9963 | 0.9932 | 0.9961 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8442 | 0.9755 | 0.8816 | 0.8625 | 0.9666 | 0.9865 | 0.9322 | 1014 |
| other | 0.8612 | 0.8849 | 0.5770 | 0.6910 | 0.9722 | 0.9494 | 0.8389 | 879 |
| stroma | 0.5296 | 0.9818 | 0.8171 | 0.6427 | 0.9315 | 0.9398 | 0.7746 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=27 TN=5575 FN=16
- **cancer**: TP=1070 FP=135 TN=4238 FN=259
- **empty**: TP=1605 FP=6 TN=4075 FN=16
- **normal**: TP=856 FP=115 TN=4573 FN=158
- **other**: TP=757 FP=555 TN=4268 FN=122
- **stroma**: TP=402 FP=90 TN=4853 FN=357

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 2 | 2 | 2 | 7 | 3 |
| cancer | 15 | 1070 | 0 | 21 | 216 | 7 |
| empty | 0 | 2 | 1605 | 0 | 12 | 2 |
| normal | 2 | 20 | 0 | 856 | 107 | 29 |
| other | 4 | 52 | 4 | 13 | 757 | 49 |
| stroma | 6 | 59 | 0 | 79 | 213 | 402 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7568    0.8400    0.7962       100
      cancer     0.8880    0.8051    0.8445      1329
       empty     0.9963    0.9901    0.9932      1621
      normal     0.8816    0.8442    0.8625      1014
       other     0.5770    0.8612    0.6910       879
      stroma     0.8171    0.5296    0.6427       759

    accuracy                         0.8373      5702
   macro avg     0.8194    0.8117    0.8050      5702
weighted avg     0.8579    0.8373    0.8386      5702

```
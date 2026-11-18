# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8572 |
| Accuracy | 0.8874 |
| Macro-AUROC (OvR) | 0.9803 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8894 |
| Balanced Acc | 0.8571 |
| Macro-Sensitivity | 0.8571 |
| Macro-Specificity | 0.9777 |
| Macro-Precision | 0.8589 |
| AUC-PR (Macro) | 0.9070 |
| Cohen's Kappa | 0.8580 |
| MCC | 0.8584 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 918.4 (1.09 ms/张) |
| End-to-End FPS | 382.3 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8000 | 0.9970 | 0.8247 | 0.8122 | 0.9964 | 0.9962 | 0.9055 | 100 |
| cancer | 0.8736 | 0.9831 | 0.9401 | 0.9056 | 0.9624 | 0.9886 | 0.9670 | 1329 |
| empty | 0.9907 | 0.9980 | 0.9950 | 0.9929 | 0.9963 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9201 | 0.9906 | 0.9550 | 0.9372 | 0.9829 | 0.9936 | 0.9800 | 1014 |
| other | 0.8111 | 0.9428 | 0.7209 | 0.7634 | 0.9648 | 0.9490 | 0.8621 | 879 |
| stroma | 0.7470 | 0.9549 | 0.7177 | 0.7321 | 0.9609 | 0.9544 | 0.7278 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=80 FP=17 TN=5585 FN=20
- **cancer**: TP=1161 FP=74 TN=4299 FN=168
- **empty**: TP=1606 FP=8 TN=4073 FN=15
- **normal**: TP=933 FP=44 TN=4644 FN=81
- **other**: TP=713 FP=276 TN=4547 FN=166
- **stroma**: TP=567 FP=223 TN=4720 FN=192

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 80 | 3 | 0 | 3 | 11 | 3 |
| cancer | 9 | 1161 | 0 | 13 | 66 | 80 |
| empty | 1 | 2 | 1606 | 0 | 9 | 3 |
| normal | 2 | 9 | 0 | 933 | 43 | 27 |
| other | 3 | 37 | 8 | 8 | 713 | 110 |
| stroma | 2 | 23 | 0 | 20 | 147 | 567 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8247    0.8000    0.8122       100
      cancer     0.9401    0.8736    0.9056      1329
       empty     0.9950    0.9907    0.9929      1621
      normal     0.9550    0.9201    0.9372      1014
       other     0.7209    0.8111    0.7634       879
      stroma     0.7177    0.7470    0.7321       759

    accuracy                         0.8874      5702
   macro avg     0.8589    0.8571    0.8572      5702
weighted avg     0.8929    0.8874    0.8894      5702

```
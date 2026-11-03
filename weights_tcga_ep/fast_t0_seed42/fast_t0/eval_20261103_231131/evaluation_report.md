# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8336 |
| Accuracy | 0.8650 |
| Macro-AUROC (OvR) | 0.9776 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8656 |
| Balanced Acc | 0.8515 |
| Macro-Sensitivity | 0.8515 |
| Macro-Specificity | 0.9734 |
| Macro-Precision | 0.8248 |
| AUC-PR (Macro) | 0.9031 |
| Cohen's Kappa | 0.8304 |
| MCC | 0.8322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 11409.8 (0.09 ms/张) |
| End-to-End FPS | 602.0 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9941 | 0.7179 | 0.7742 | 0.9971 | 0.9963 | 0.8916 | 100 |
| cancer | 0.7494 | 0.9863 | 0.9432 | 0.8352 | 0.9283 | 0.9823 | 0.9482 | 1329 |
| empty | 0.9944 | 0.9961 | 0.9902 | 0.9923 | 0.9978 | 0.9999 | 0.9996 | 1621 |
| normal | 0.9379 | 0.9770 | 0.8980 | 0.9175 | 0.9864 | 0.9931 | 0.9722 | 1014 |
| other | 0.7042 | 0.9451 | 0.7002 | 0.7022 | 0.9460 | 0.9263 | 0.8008 | 879 |
| stroma | 0.8827 | 0.9417 | 0.6994 | 0.7804 | 0.9812 | 0.9678 | 0.8060 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=33 TN=5569 FN=16
- **cancer**: TP=996 FP=60 TN=4313 FN=333
- **empty**: TP=1612 FP=16 TN=4065 FN=9
- **normal**: TP=951 FP=108 TN=4580 FN=63
- **other**: TP=619 FP=265 TN=4558 FN=260
- **stroma**: TP=670 FP=288 TN=4655 FN=89

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 1 | 2 | 3 | 5 | 5 |
| cancer | 14 | 996 | 0 | 47 | 191 | 81 |
| empty | 1 | 0 | 1612 | 1 | 6 | 1 |
| normal | 8 | 6 | 0 | 951 | 29 | 20 |
| other | 3 | 39 | 14 | 23 | 619 | 181 |
| stroma | 7 | 14 | 0 | 34 | 34 | 670 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7179    0.8400    0.7742       100
      cancer     0.9432    0.7494    0.8352      1329
       empty     0.9902    0.9944    0.9923      1621
      normal     0.8980    0.9379    0.9175      1014
       other     0.7002    0.7042    0.7022       879
      stroma     0.6994    0.8827    0.7804       759

    accuracy                         0.8650      5702
   macro avg     0.8248    0.8515    0.8336      5702
weighted avg     0.8747    0.8650    0.8656      5702

```
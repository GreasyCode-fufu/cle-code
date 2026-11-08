# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7855 |
| Accuracy | 0.8280 |
| Macro-AUROC (OvR) | 0.9424 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8231 |
| Balanced Acc | 0.7881 |
| Macro-Sensitivity | 0.7881 |
| Macro-Specificity | 0.9642 |
| Macro-Precision | 0.7894 |
| AUC-PR (Macro) | 0.8384 |
| Cohen's Kappa | 0.7782 |
| MCC | 0.7797 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9597.1 (0.10 ms/张) |
| End-to-End FPS | 554.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8442 | 0.9947 | 0.7534 | 0.7962 | 0.9970 | 0.9955 | 0.9075 | 199 |
| cancer | 0.9104 | 0.9372 | 0.8275 | 0.8670 | 0.9693 | 0.9777 | 0.9299 | 2645 |
| empty | 0.9975 | 0.9611 | 0.9188 | 0.9566 | 0.9989 | 0.9825 | 0.9378 | 3256 |
| normal | 0.7193 | 0.9818 | 0.8851 | 0.7936 | 0.9472 | 0.9181 | 0.8530 | 1735 |
| other | 0.5871 | 0.9498 | 0.6923 | 0.6354 | 0.9228 | 0.8440 | 0.6918 | 1717 |
| stroma | 0.6701 | 0.9607 | 0.6591 | 0.6645 | 0.9626 | 0.9365 | 0.7105 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=168 FP=55 TN=10380 FN=31
- **cancer**: TP=2408 FP=502 TN=7487 FN=237
- **empty**: TP=3248 FP=287 TN=7091 FN=8
- **normal**: TP=1248 FP=162 TN=8737 FN=487
- **other**: TP=1008 FP=448 TN=8469 FN=709
- **stroma**: TP=725 FP=375 TN=9177 FN=357

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 168 | 5 | 2 | 9 | 12 | 3 |
| cancer | 9 | 2408 | 1 | 48 | 99 | 80 |
| empty | 0 | 0 | 3248 | 0 | 8 | 0 |
| normal | 0 | 192 | 1 | 1248 | 159 | 135 |
| other | 42 | 205 | 283 | 22 | 1008 | 157 |
| stroma | 4 | 100 | 0 | 83 | 170 | 725 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7534    0.8442    0.7962       199
      cancer     0.8275    0.9104    0.8670      2645
       empty     0.9188    0.9975    0.9566      3256
      normal     0.8851    0.7193    0.7936      1735
       other     0.6923    0.5871    0.6354      1717
      stroma     0.6591    0.6701    0.6645      1082

    accuracy                         0.8280     10634
   macro avg     0.7894    0.7881    0.7855     10634
weighted avg     0.8245    0.8280    0.8231     10634

```
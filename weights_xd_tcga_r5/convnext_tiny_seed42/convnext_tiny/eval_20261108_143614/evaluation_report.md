# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8067 |
| Accuracy | 0.8386 |
| Macro-AUROC (OvR) | 0.9558 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8351 |
| Balanced Acc | 0.8034 |
| Macro-Sensitivity | 0.8034 |
| Macro-Specificity | 0.9663 |
| Macro-Precision | 0.8129 |
| AUC-PR (Macro) | 0.8621 |
| Cohen's Kappa | 0.7919 |
| MCC | 0.7927 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 649.4 (1.54 ms/张) |
| End-to-End FPS | 328.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8643 | 0.9964 | 0.8190 | 0.8411 | 0.9974 | 0.9976 | 0.9304 | 199 |
| cancer | 0.8904 | 0.9448 | 0.8423 | 0.8656 | 0.9630 | 0.9715 | 0.9156 | 2645 |
| empty | 0.9954 | 0.9610 | 0.9184 | 0.9553 | 0.9979 | 0.9755 | 0.9113 | 3256 |
| normal | 0.7712 | 0.9791 | 0.8780 | 0.8211 | 0.9564 | 0.9669 | 0.9055 | 1735 |
| other | 0.6395 | 0.9433 | 0.6845 | 0.6612 | 0.9315 | 0.8632 | 0.7337 | 1717 |
| stroma | 0.6599 | 0.9731 | 0.7353 | 0.6956 | 0.9619 | 0.9601 | 0.7761 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=172 FP=38 TN=10397 FN=27
- **cancer**: TP=2355 FP=441 TN=7548 FN=290
- **empty**: TP=3241 FP=288 TN=7090 FN=15
- **normal**: TP=1338 FP=186 TN=8713 FN=397
- **other**: TP=1098 FP=506 TN=8411 FN=619
- **stroma**: TP=714 FP=257 TN=9295 FN=368

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 172 | 7 | 2 | 6 | 10 | 2 |
| cancer | 2 | 2355 | 1 | 78 | 172 | 37 |
| empty | 0 | 0 | 3241 | 0 | 15 | 0 |
| normal | 1 | 176 | 2 | 1338 | 101 | 117 |
| other | 32 | 169 | 283 | 34 | 1098 | 101 |
| stroma | 3 | 89 | 0 | 68 | 208 | 714 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8190    0.8643    0.8411       199
      cancer     0.8423    0.8904    0.8656      2645
       empty     0.9184    0.9954    0.9553      3256
      normal     0.8780    0.7712    0.8211      1735
       other     0.6845    0.6395    0.6612      1717
      stroma     0.7353    0.6599    0.6956      1082

    accuracy                         0.8386     10634
   macro avg     0.8129    0.8034    0.8067     10634
weighted avg     0.8346    0.8386    0.8351     10634

```
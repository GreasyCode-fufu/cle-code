# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7870 |
| Accuracy | 0.8219 |
| Macro-AUROC (OvR) | 0.9616 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8178 |
| Balanced Acc | 0.7891 |
| Macro-Sensitivity | 0.7891 |
| Macro-Specificity | 0.9632 |
| Macro-Precision | 0.7919 |
| AUC-PR (Macro) | 0.8505 |
| Cohen's Kappa | 0.7708 |
| MCC | 0.7723 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 285.3 (3.51 ms/张) |
| End-to-End FPS | 199.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9966 | 0.8333 | 0.8557 | 0.9977 | 0.9978 | 0.9340 | 199 |
| cancer | 0.8832 | 0.9390 | 0.8275 | 0.8544 | 0.9604 | 0.9738 | 0.9437 | 2645 |
| empty | 0.9963 | 0.9612 | 0.9190 | 0.9561 | 0.9983 | 0.9769 | 0.8960 | 3256 |
| normal | 0.7625 | 0.9813 | 0.8885 | 0.8207 | 0.9549 | 0.9800 | 0.9241 | 1735 |
| other | 0.5504 | 0.9566 | 0.7095 | 0.6199 | 0.9170 | 0.8958 | 0.7130 | 1717 |
| stroma | 0.6627 | 0.9442 | 0.5736 | 0.6149 | 0.9611 | 0.9455 | 0.6923 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=35 TN=10400 FN=24
- **cancer**: TP=2336 FP=487 TN=7502 FN=309
- **empty**: TP=3244 FP=286 TN=7092 FN=12
- **normal**: TP=1323 FP=166 TN=8733 FN=412
- **other**: TP=945 FP=387 TN=8530 FN=772
- **stroma**: TP=717 FP=533 TN=9019 FN=365

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 6 | 2 | 4 | 10 | 2 |
| cancer | 2 | 2336 | 0 | 95 | 73 | 139 |
| empty | 0 | 0 | 3244 | 0 | 12 | 0 |
| normal | 5 | 208 | 3 | 1323 | 76 | 120 |
| other | 25 | 172 | 281 | 22 | 945 | 272 |
| stroma | 3 | 101 | 0 | 45 | 216 | 717 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8333    0.8794    0.8557       199
      cancer     0.8275    0.8832    0.8544      2645
       empty     0.9190    0.9963    0.9561      3256
      normal     0.8885    0.7625    0.8207      1735
       other     0.7095    0.5504    0.6199      1717
      stroma     0.5736    0.6627    0.6149      1082

    accuracy                         0.8219     10634
   macro avg     0.7919    0.7891    0.7870     10634
weighted avg     0.8207    0.8219    0.8178     10634

```
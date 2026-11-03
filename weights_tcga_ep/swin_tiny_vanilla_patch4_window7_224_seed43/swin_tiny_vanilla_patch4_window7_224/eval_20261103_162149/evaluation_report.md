# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8029 |
| Accuracy | 0.8394 |
| Macro-AUROC (OvR) | 0.9571 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8364 |
| Balanced Acc | 0.8087 |
| Macro-Sensitivity | 0.8087 |
| Macro-Specificity | 0.9669 |
| Macro-Precision | 0.8054 |
| AUC-PR (Macro) | 0.8567 |
| Cohen's Kappa | 0.7933 |
| MCC | 0.7947 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 643.7 (1.55 ms/张) |
| End-to-End FPS | 322.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8894 | 0.9951 | 0.7763 | 0.8290 | 0.9979 | 0.9950 | 0.9034 | 199 |
| cancer | 0.9270 | 0.9599 | 0.8846 | 0.9053 | 0.9755 | 0.9832 | 0.9624 | 2645 |
| empty | 0.9957 | 0.9623 | 0.9210 | 0.9569 | 0.9980 | 0.9844 | 0.9631 | 3256 |
| normal | 0.6749 | 0.9880 | 0.9163 | 0.7773 | 0.9397 | 0.9414 | 0.8666 | 1735 |
| other | 0.6412 | 0.9343 | 0.6526 | 0.6469 | 0.9312 | 0.8974 | 0.6732 | 1717 |
| stroma | 0.7237 | 0.9617 | 0.6815 | 0.7019 | 0.9685 | 0.9412 | 0.7715 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=177 FP=51 TN=10384 FN=22
- **cancer**: TP=2452 FP=320 TN=7669 FN=193
- **empty**: TP=3242 FP=278 TN=7100 FN=14
- **normal**: TP=1171 FP=107 TN=8792 FN=564
- **other**: TP=1101 FP=586 TN=8331 FN=616
- **stroma**: TP=783 FP=366 TN=9186 FN=299

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 177 | 5 | 2 | 0 | 12 | 3 |
| cancer | 6 | 2452 | 2 | 46 | 68 | 71 |
| empty | 0 | 2 | 3242 | 0 | 12 | 0 |
| normal | 2 | 124 | 0 | 1171 | 289 | 149 |
| other | 41 | 131 | 274 | 27 | 1101 | 143 |
| stroma | 2 | 58 | 0 | 34 | 205 | 783 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7763    0.8894    0.8290       199
      cancer     0.8846    0.9270    0.9053      2645
       empty     0.9210    0.9957    0.9569      3256
      normal     0.9163    0.6749    0.7773      1735
       other     0.6526    0.6412    0.6469      1717
      stroma     0.6815    0.7237    0.7019      1082

    accuracy                         0.8394     10634
   macro avg     0.8054    0.8087    0.8029     10634
weighted avg     0.8408    0.8394    0.8364     10634

```
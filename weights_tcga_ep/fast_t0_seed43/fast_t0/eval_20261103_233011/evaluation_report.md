# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7863 |
| Accuracy | 0.8259 |
| Macro-AUROC (OvR) | 0.9634 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8243 |
| Balanced Acc | 0.7846 |
| Macro-Sensitivity | 0.7846 |
| Macro-Specificity | 0.9644 |
| Macro-Precision | 0.7920 |
| AUC-PR (Macro) | 0.8569 |
| Cohen's Kappa | 0.7762 |
| MCC | 0.7770 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 10741.8 (0.09 ms/张) |
| End-to-End FPS | 602.0 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8492 | 0.9963 | 0.8125 | 0.8305 | 0.9971 | 0.9968 | 0.9130 | 199 |
| cancer | 0.9153 | 0.9697 | 0.9091 | 0.9122 | 0.9719 | 0.9896 | 0.9734 | 2645 |
| empty | 0.9899 | 0.9648 | 0.9254 | 0.9565 | 0.9954 | 0.9977 | 0.9948 | 3256 |
| normal | 0.6847 | 0.9769 | 0.8522 | 0.7593 | 0.9408 | 0.9435 | 0.8392 | 1735 |
| other | 0.6447 | 0.9156 | 0.5952 | 0.6190 | 0.9305 | 0.9086 | 0.7072 | 1717 |
| stroma | 0.6238 | 0.9633 | 0.6579 | 0.6404 | 0.9576 | 0.9439 | 0.7137 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=169 FP=39 TN=10396 FN=30
- **cancer**: TP=2421 FP=242 TN=7747 FN=224
- **empty**: TP=3223 FP=260 TN=7118 FN=33
- **normal**: TP=1188 FP=206 TN=8693 FN=547
- **other**: TP=1107 FP=753 TN=8164 FN=610
- **stroma**: TP=675 FP=351 TN=9201 FN=407

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 169 | 4 | 3 | 4 | 15 | 4 |
| cancer | 4 | 2421 | 1 | 118 | 66 | 35 |
| empty | 0 | 8 | 3223 | 0 | 25 | 0 |
| normal | 0 | 63 | 1 | 1188 | 318 | 165 |
| other | 34 | 128 | 255 | 46 | 1107 | 147 |
| stroma | 1 | 39 | 0 | 38 | 329 | 675 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8125    0.8492    0.8305       199
      cancer     0.9091    0.9153    0.9122      2645
       empty     0.9254    0.9899    0.9565      3256
      normal     0.8522    0.6847    0.7593      1735
       other     0.5952    0.6447    0.6190      1717
      stroma     0.6579    0.6238    0.6404      1082

    accuracy                         0.8259     10634
   macro avg     0.7920    0.7846    0.7863     10634
weighted avg     0.8267    0.8259    0.8243     10634

```
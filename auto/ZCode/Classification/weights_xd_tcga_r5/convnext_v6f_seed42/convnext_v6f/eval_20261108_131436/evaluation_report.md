# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8000 |
| Accuracy | 0.8430 |
| Macro-AUROC (OvR) | 0.9542 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8379 |
| Balanced Acc | 0.8002 |
| Macro-Sensitivity | 0.8002 |
| Macro-Specificity | 0.9670 |
| Macro-Precision | 0.8063 |
| AUC-PR (Macro) | 0.8546 |
| Cohen's Kappa | 0.7972 |
| MCC | 0.7987 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 613.8 (1.63 ms/张) |
| End-to-End FPS | 320.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8141 | 0.9937 | 0.7105 | 0.7588 | 0.9964 | 0.9971 | 0.8816 | 199 |
| cancer | 0.9149 | 0.9329 | 0.8187 | 0.8641 | 0.9707 | 0.9637 | 0.8679 | 2645 |
| empty | 0.9942 | 0.9627 | 0.9217 | 0.9566 | 0.9973 | 0.9835 | 0.9598 | 3256 |
| normal | 0.7827 | 0.9730 | 0.8498 | 0.8149 | 0.9583 | 0.9535 | 0.8792 | 1735 |
| other | 0.6069 | 0.9597 | 0.7438 | 0.6684 | 0.9269 | 0.8707 | 0.7350 | 1717 |
| stroma | 0.6885 | 0.9797 | 0.7934 | 0.7373 | 0.9652 | 0.9564 | 0.8043 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=162 FP=66 TN=10369 FN=37
- **cancer**: TP=2420 FP=536 TN=7453 FN=225
- **empty**: TP=3237 FP=275 TN=7103 FN=19
- **normal**: TP=1358 FP=240 TN=8659 FN=377
- **other**: TP=1042 FP=359 TN=8558 FN=675
- **stroma**: TP=745 FP=194 TN=9358 FN=337

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 162 | 13 | 2 | 7 | 14 | 1 |
| cancer | 9 | 2420 | 0 | 102 | 95 | 19 |
| empty | 0 | 0 | 3237 | 0 | 19 | 0 |
| normal | 2 | 196 | 0 | 1358 | 110 | 69 |
| other | 51 | 216 | 273 | 30 | 1042 | 105 |
| stroma | 4 | 111 | 0 | 101 | 121 | 745 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7105    0.8141    0.7588       199
      cancer     0.8187    0.9149    0.8641      2645
       empty     0.9217    0.9942    0.9566      3256
      normal     0.8498    0.7827    0.8149      1735
       other     0.7438    0.6069    0.6684      1717
      stroma     0.7934    0.6885    0.7373      1082

    accuracy                         0.8430     10634
   macro avg     0.8063    0.8002    0.8000     10634
weighted avg     0.8386    0.8430    0.8379     10634

```
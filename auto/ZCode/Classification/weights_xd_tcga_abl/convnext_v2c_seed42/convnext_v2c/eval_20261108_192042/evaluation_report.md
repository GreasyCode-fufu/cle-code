# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8072 |
| Accuracy | 0.8435 |
| Macro-AUROC (OvR) | 0.9670 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8378 |
| Balanced Acc | 0.8050 |
| Macro-Sensitivity | 0.8050 |
| Macro-Specificity | 0.9673 |
| Macro-Precision | 0.8172 |
| AUC-PR (Macro) | 0.8778 |
| Cohen's Kappa | 0.7982 |
| MCC | 0.8001 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 888.6 (1.13 ms/张) |
| End-to-End FPS | 380.1 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8191 | 0.9966 | 0.8232 | 0.8212 | 0.9966 | 0.9975 | 0.9172 | 199 |
| cancer | 0.9263 | 0.9418 | 0.8405 | 0.8813 | 0.9747 | 0.9812 | 0.9550 | 2645 |
| empty | 0.9954 | 0.9595 | 0.9155 | 0.9538 | 0.9979 | 0.9826 | 0.9574 | 3256 |
| normal | 0.7706 | 0.9824 | 0.8949 | 0.8281 | 0.9565 | 0.9648 | 0.9015 | 1735 |
| other | 0.5550 | 0.9620 | 0.7376 | 0.6334 | 0.9182 | 0.9107 | 0.7368 | 1717 |
| stroma | 0.7634 | 0.9614 | 0.6912 | 0.7255 | 0.9729 | 0.9654 | 0.7988 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=163 FP=35 TN=10400 FN=36
- **cancer**: TP=2450 FP=465 TN=7524 FN=195
- **empty**: TP=3241 FP=299 TN=7079 FN=15
- **normal**: TP=1337 FP=157 TN=8742 FN=398
- **other**: TP=953 FP=339 TN=8578 FN=764
- **stroma**: TP=826 FP=369 TN=9183 FN=256

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 163 | 9 | 2 | 6 | 16 | 3 |
| cancer | 4 | 2450 | 2 | 45 | 62 | 82 |
| empty | 0 | 5 | 3241 | 0 | 10 | 0 |
| normal | 1 | 155 | 2 | 1337 | 162 | 78 |
| other | 28 | 193 | 293 | 44 | 953 | 206 |
| stroma | 2 | 103 | 0 | 62 | 89 | 826 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8232    0.8191    0.8212       199
      cancer     0.8405    0.9263    0.8813      2645
       empty     0.9155    0.9954    0.9538      3256
      normal     0.8949    0.7706    0.8281      1735
       other     0.7376    0.5550    0.6334      1717
      stroma     0.6912    0.7634    0.7255      1082

    accuracy                         0.8435     10634
   macro avg     0.8172    0.8050    0.8072     10634
weighted avg     0.8402    0.8435    0.8378     10634

```
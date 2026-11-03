# cle_convnext_sn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/cle_convnext_sn_seed44/cle_convnext_sn/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8101 |
| Accuracy | 0.8476 |
| Macro-AUROC (OvR) | 0.9660 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8399 |
| Balanced Acc | 0.7975 |
| Macro-Sensitivity | 0.7975 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.8343 |
| AUC-PR (Macro) | 0.8780 |
| Cohen's Kappa | 0.8028 |
| MCC | 0.8052 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 619.3 (1.61 ms/张) |
| End-to-End FPS | 321.0 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7990 | 0.9978 | 0.8736 | 0.8346 | 0.9962 | 0.9950 | 0.9205 | 199 |
| cancer | 0.9323 | 0.9302 | 0.8155 | 0.8700 | 0.9765 | 0.9834 | 0.9565 | 2645 |
| empty | 0.9939 | 0.9642 | 0.9246 | 0.9580 | 0.9972 | 0.9794 | 0.9337 | 3256 |
| normal | 0.8697 | 0.9630 | 0.8210 | 0.8447 | 0.9743 | 0.9807 | 0.9320 | 1735 |
| other | 0.5591 | 0.9712 | 0.7888 | 0.6544 | 0.9196 | 0.8965 | 0.7489 | 1717 |
| stroma | 0.6312 | 0.9801 | 0.7824 | 0.6987 | 0.9591 | 0.9608 | 0.7766 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=159 FP=23 TN=10412 FN=40
- **cancer**: TP=2466 FP=558 TN=7431 FN=179
- **empty**: TP=3236 FP=264 TN=7114 FN=20
- **normal**: TP=1509 FP=329 TN=8570 FN=226
- **other**: TP=960 FP=257 TN=8660 FN=757
- **stroma**: TP=683 FP=190 TN=9362 FN=399

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 159 | 19 | 2 | 8 | 10 | 1 |
| cancer | 2 | 2466 | 1 | 102 | 65 | 9 |
| empty | 1 | 2 | 3236 | 0 | 17 | 0 |
| normal | 0 | 132 | 0 | 1509 | 31 | 63 |
| other | 19 | 301 | 261 | 59 | 960 | 117 |
| stroma | 1 | 104 | 0 | 160 | 134 | 683 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8736    0.7990    0.8346       199
      cancer     0.8155    0.9323    0.8700      2645
       empty     0.9246    0.9939    0.9580      3256
      normal     0.8210    0.8697    0.8447      1735
       other     0.7888    0.5591    0.6544      1717
      stroma     0.7824    0.6312    0.6987      1082

    accuracy                         0.8476     10634
   macro avg     0.8343    0.7975    0.8101     10634
weighted avg     0.8432    0.8476    0.8399     10634

```
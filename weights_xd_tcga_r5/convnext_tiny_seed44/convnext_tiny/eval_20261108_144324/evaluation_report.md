# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8117 |
| Accuracy | 0.8461 |
| Macro-AUROC (OvR) | 0.9667 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8390 |
| Balanced Acc | 0.8044 |
| Macro-Sensitivity | 0.8044 |
| Macro-Specificity | 0.9675 |
| Macro-Precision | 0.8279 |
| AUC-PR (Macro) | 0.8823 |
| Cohen's Kappa | 0.8010 |
| MCC | 0.8032 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 923.2 (1.08 ms/张) |
| End-to-End FPS | 386.4 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8492 | 0.9971 | 0.8492 | 0.8492 | 0.9971 | 0.9956 | 0.9296 | 199 |
| cancer | 0.9361 | 0.9287 | 0.8129 | 0.8701 | 0.9777 | 0.9844 | 0.9596 | 2645 |
| empty | 0.9936 | 0.9642 | 0.9245 | 0.9578 | 0.9971 | 0.9816 | 0.9496 | 3256 |
| normal | 0.8334 | 0.9690 | 0.8397 | 0.8366 | 0.9676 | 0.9754 | 0.9204 | 1735 |
| other | 0.5626 | 0.9679 | 0.7716 | 0.6507 | 0.9200 | 0.8999 | 0.7474 | 1717 |
| stroma | 0.6516 | 0.9779 | 0.7697 | 0.7057 | 0.9612 | 0.9633 | 0.7869 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=169 FP=30 TN=10405 FN=30
- **cancer**: TP=2476 FP=570 TN=7419 FN=169
- **empty**: TP=3235 FP=264 TN=7114 FN=21
- **normal**: TP=1446 FP=276 TN=8623 FN=289
- **other**: TP=966 FP=286 TN=8631 FN=751
- **stroma**: TP=705 FP=211 TN=9341 FN=377

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 169 | 12 | 2 | 5 | 8 | 3 |
| cancer | 3 | 2476 | 1 | 86 | 66 | 13 |
| empty | 0 | 2 | 3235 | 0 | 18 | 1 |
| normal | 0 | 164 | 0 | 1446 | 61 | 64 |
| other | 26 | 282 | 261 | 52 | 966 | 130 |
| stroma | 1 | 110 | 0 | 133 | 133 | 705 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8492    0.8492    0.8492       199
      cancer     0.8129    0.9361    0.8701      2645
       empty     0.9245    0.9936    0.9578      3256
      normal     0.8397    0.8334    0.8366      1735
       other     0.7716    0.5626    0.6507      1717
      stroma     0.7697    0.6516    0.7057      1082

    accuracy                         0.8461     10634
   macro avg     0.8279    0.8044    0.8117     10634
weighted avg     0.8411    0.8461    0.8390     10634

```
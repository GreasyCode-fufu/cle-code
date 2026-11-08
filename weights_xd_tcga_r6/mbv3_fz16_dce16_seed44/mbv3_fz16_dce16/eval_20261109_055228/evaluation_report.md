# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7890 |
| Accuracy | 0.8328 |
| Macro-AUROC (OvR) | 0.9468 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8287 |
| Balanced Acc | 0.7856 |
| Macro-Sensitivity | 0.7856 |
| Macro-Specificity | 0.9653 |
| Macro-Precision | 0.7972 |
| AUC-PR (Macro) | 0.8394 |
| Cohen's Kappa | 0.7843 |
| MCC | 0.7853 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9618.4 (0.10 ms/张) |
| End-to-End FPS | 559.8 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9956 | 0.7840 | 0.8107 | 0.9969 | 0.9959 | 0.8977 | 199 |
| cancer | 0.9229 | 0.9526 | 0.8656 | 0.8933 | 0.9739 | 0.9862 | 0.9625 | 2645 |
| empty | 0.9936 | 0.9596 | 0.9157 | 0.9530 | 0.9970 | 0.9838 | 0.9458 | 3256 |
| normal | 0.7493 | 0.9827 | 0.8941 | 0.8153 | 0.9526 | 0.9375 | 0.8748 | 1735 |
| other | 0.6377 | 0.9310 | 0.6404 | 0.6390 | 0.9303 | 0.8599 | 0.6694 | 1717 |
| stroma | 0.5712 | 0.9701 | 0.6836 | 0.6224 | 0.9523 | 0.9174 | 0.6862 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=46 TN=10389 FN=32
- **cancer**: TP=2441 FP=379 TN=7610 FN=204
- **empty**: TP=3235 FP=298 TN=7080 FN=21
- **normal**: TP=1300 FP=154 TN=8745 FN=435
- **other**: TP=1095 FP=615 TN=8302 FN=622
- **stroma**: TP=618 FP=286 TN=9266 FN=464

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 6 | 4 | 5 | 15 | 2 |
| cancer | 3 | 2441 | 1 | 40 | 108 | 52 |
| empty | 0 | 0 | 3235 | 0 | 21 | 0 |
| normal | 2 | 150 | 5 | 1300 | 184 | 94 |
| other | 38 | 142 | 287 | 17 | 1095 | 138 |
| stroma | 3 | 81 | 1 | 92 | 287 | 618 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7840    0.8392    0.8107       199
      cancer     0.8656    0.9229    0.8933      2645
       empty     0.9157    0.9936    0.9530      3256
      normal     0.8941    0.7493    0.8153      1735
       other     0.6404    0.6377    0.6390      1717
      stroma     0.6836    0.5712    0.6224      1082

    accuracy                         0.8328     10634
   macro avg     0.7972    0.7856    0.7890     10634
weighted avg     0.8292    0.8328    0.8287     10634

```
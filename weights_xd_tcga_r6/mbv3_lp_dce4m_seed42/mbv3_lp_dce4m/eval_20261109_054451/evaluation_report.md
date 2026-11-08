# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_lp_dce4m_seed42/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8121 |
| Accuracy | 0.8518 |
| Macro-AUROC (OvR) | 0.9752 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8544 |
| Balanced Acc | 0.8080 |
| Macro-Sensitivity | 0.8080 |
| Macro-Specificity | 0.9704 |
| Macro-Precision | 0.8209 |
| AUC-PR (Macro) | 0.8809 |
| Cohen's Kappa | 0.8131 |
| MCC | 0.8140 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 6952.9 (0.14 ms/张) |
| End-to-End FPS | 22.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.6900 | 0.9961 | 0.7582 | 0.7225 | 0.9945 | 0.9900 | 0.8092 | 100 |
| cancer | 0.8315 | 0.9614 | 0.8673 | 0.8490 | 0.9494 | 0.9692 | 0.9252 | 1329 |
| empty | 0.9932 | 0.9973 | 0.9932 | 0.9932 | 0.9973 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8077 | 0.9906 | 0.9490 | 0.8727 | 0.9597 | 0.9921 | 0.9648 | 1014 |
| other | 0.7986 | 0.9264 | 0.6641 | 0.7252 | 0.9619 | 0.9465 | 0.8400 | 879 |
| stroma | 0.7273 | 0.9506 | 0.6935 | 0.7100 | 0.9578 | 0.9538 | 0.7464 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=69 FP=22 TN=5580 FN=31
- **cancer**: TP=1105 FP=169 TN=4204 FN=224
- **empty**: TP=1610 FP=11 TN=4070 FN=11
- **normal**: TP=819 FP=44 TN=4644 FN=195
- **other**: TP=702 FP=355 TN=4468 FN=177
- **stroma**: TP=552 FP=244 TN=4699 FN=207

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 69 | 5 | 0 | 4 | 17 | 5 |
| cancer | 19 | 1105 | 0 | 21 | 129 | 55 |
| empty | 0 | 0 | 1610 | 0 | 9 | 2 |
| normal | 1 | 88 | 0 | 819 | 38 | 68 |
| other | 2 | 46 | 11 | 4 | 702 | 114 |
| stroma | 0 | 30 | 0 | 15 | 162 | 552 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7582    0.6900    0.7225       100
      cancer     0.8673    0.8315    0.8490      1329
       empty     0.9932    0.9932    0.9932      1621
      normal     0.9490    0.8077    0.8727      1014
       other     0.6641    0.7986    0.7252       879
      stroma     0.6935    0.7273    0.7100       759

    accuracy                         0.8518      5702
   macro avg     0.8209    0.8080    0.8121      5702
weighted avg     0.8613    0.8518    0.8544      5702

```
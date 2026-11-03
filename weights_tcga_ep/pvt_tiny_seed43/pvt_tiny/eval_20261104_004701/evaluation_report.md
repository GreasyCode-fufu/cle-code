# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7868 |
| Accuracy | 0.8352 |
| Macro-AUROC (OvR) | 0.9597 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8295 |
| Balanced Acc | 0.7857 |
| Macro-Sensitivity | 0.7857 |
| Macro-Specificity | 0.9653 |
| Macro-Precision | 0.8000 |
| AUC-PR (Macro) | 0.8494 |
| Cohen's Kappa | 0.7867 |
| MCC | 0.7890 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 1173.2 (0.85 ms/张) |
| End-to-End FPS | 422.6 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8342 | 0.9935 | 0.7094 | 0.7667 | 0.9968 | 0.9928 | 0.8435 | 199 |
| cancer | 0.9414 | 0.9249 | 0.8058 | 0.8684 | 0.9795 | 0.9812 | 0.9567 | 2645 |
| empty | 0.9966 | 0.9627 | 0.9219 | 0.9578 | 0.9985 | 0.9751 | 0.9083 | 3256 |
| normal | 0.7159 | 0.9873 | 0.9166 | 0.8039 | 0.9469 | 0.9694 | 0.8989 | 1735 |
| other | 0.6476 | 0.9448 | 0.6933 | 0.6697 | 0.9330 | 0.8903 | 0.7415 | 1717 |
| stroma | 0.5786 | 0.9785 | 0.7533 | 0.6545 | 0.9535 | 0.9491 | 0.7472 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=166 FP=68 TN=10367 FN=33
- **cancer**: TP=2490 FP=600 TN=7389 FN=155
- **empty**: TP=3245 FP=275 TN=7103 FN=11
- **normal**: TP=1242 FP=113 TN=8786 FN=493
- **other**: TP=1112 FP=492 TN=8425 FN=605
- **stroma**: TP=626 FP=205 TN=9347 FN=456

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 166 | 5 | 2 | 3 | 22 | 1 |
| cancer | 13 | 2490 | 2 | 46 | 71 | 23 |
| empty | 0 | 2 | 3245 | 0 | 9 | 0 |
| normal | 1 | 248 | 2 | 1242 | 129 | 113 |
| other | 53 | 209 | 267 | 8 | 1112 | 68 |
| stroma | 1 | 136 | 2 | 56 | 261 | 626 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7094    0.8342    0.7667       199
      cancer     0.8058    0.9414    0.8684      2645
       empty     0.9219    0.9966    0.9578      3256
      normal     0.9166    0.7159    0.8039      1735
       other     0.6933    0.6476    0.6697      1717
      stroma     0.7533    0.5786    0.6545      1082

    accuracy                         0.8352     10634
   macro avg     0.8000    0.7857    0.7868     10634
weighted avg     0.8341    0.8352    0.8295     10634

```
# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7995 |
| Accuracy | 0.8404 |
| Macro-AUROC (OvR) | 0.9609 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8368 |
| Balanced Acc | 0.7970 |
| Macro-Sensitivity | 0.7970 |
| Macro-Specificity | 0.9668 |
| Macro-Precision | 0.8067 |
| AUC-PR (Macro) | 0.8599 |
| Cohen's Kappa | 0.7944 |
| MCC | 0.7955 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 7839.4 (0.13 ms/张) |
| End-to-End FPS | 205.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7990 | 0.9955 | 0.7718 | 0.7852 | 0.9962 | 0.9908 | 0.8612 | 199 |
| cancer | 0.9180 | 0.9459 | 0.8490 | 0.8821 | 0.9721 | 0.9852 | 0.9570 | 2645 |
| empty | 0.9929 | 0.9623 | 0.9208 | 0.9555 | 0.9968 | 0.9899 | 0.9691 | 3256 |
| normal | 0.7187 | 0.9831 | 0.8926 | 0.7963 | 0.9472 | 0.9439 | 0.8638 | 1735 |
| other | 0.6389 | 0.9482 | 0.7037 | 0.6697 | 0.9317 | 0.8970 | 0.7419 | 1717 |
| stroma | 0.7144 | 0.9657 | 0.7021 | 0.7082 | 0.9676 | 0.9585 | 0.7661 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=159 FP=47 TN=10388 FN=40
- **cancer**: TP=2428 FP=432 TN=7557 FN=217
- **empty**: TP=3233 FP=278 TN=7100 FN=23
- **normal**: TP=1247 FP=150 TN=8749 FN=488
- **other**: TP=1097 FP=462 TN=8455 FN=620
- **stroma**: TP=773 FP=328 TN=9224 FN=309

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 159 | 5 | 2 | 5 | 25 | 3 |
| cancer | 8 | 2428 | 1 | 40 | 104 | 64 |
| empty | 0 | 1 | 3233 | 0 | 22 | 0 |
| normal | 1 | 181 | 2 | 1247 | 183 | 121 |
| other | 32 | 156 | 273 | 19 | 1097 | 140 |
| stroma | 6 | 89 | 0 | 86 | 128 | 773 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7718    0.7990    0.7852       199
      cancer     0.8490    0.9180    0.8821      2645
       empty     0.9208    0.9929    0.9555      3256
      normal     0.8926    0.7187    0.7963      1735
       other     0.7037    0.6389    0.6697      1717
      stroma     0.7021    0.7144    0.7082      1082

    accuracy                         0.8404     10634
   macro avg     0.8067    0.7970    0.7995     10634
weighted avg     0.8382    0.8404    0.8368     10634

```
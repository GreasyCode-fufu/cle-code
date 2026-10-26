# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9107 |
| Accuracy | 0.9115 |
| Macro-AUROC (OvR) | 0.9861 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9110 |
| Balanced Acc | 0.9099 |
| Macro-Sensitivity | 0.9099 |
| Macro-Specificity | 0.9552 |
| Macro-Precision | 0.9127 |
| AUC-PR (Macro) | 0.9732 |
| Cohen's Kappa | 0.8667 |
| MCC | 0.8672 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 4099.8 (0.24 ms/张) |
| End-to-End FPS | 260.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9087 | 0.9321 | 0.8832 | 0.8957 | 0.9475 | 0.9778 | 0.9621 | 208 |
| Malignant_lesions | 0.8523 | 0.9675 | 0.9202 | 0.8850 | 0.9370 | 0.9826 | 0.9614 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9661 | 0.9347 | 0.9514 | 0.9841 | 0.9979 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=25 TN=343 FN=19
- **Malignant_lesions**: TP=150 FP=13 TN=387 FN=26
- **body_of_stomach_normal**: TP=186 FP=13 TN=371 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 11 | 8 |
| Malignant_lesions | 21 | 150 | 5 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8832    0.9087    0.8957       208
     Malignant_lesions     0.9202    0.8523    0.8850       176
body_of_stomach_normal     0.9347    0.9688    0.9514       192

              accuracy                         0.9115       576
             macro avg     0.9127    0.9099    0.9107       576
          weighted avg     0.9117    0.9115    0.9110       576

```
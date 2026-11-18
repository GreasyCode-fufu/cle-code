# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed45/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8999 |
| Accuracy | 0.9028 |
| Macro-AUROC (OvR) | 0.9863 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9011 |
| Balanced Acc | 0.8998 |
| Macro-Sensitivity | 0.8998 |
| Macro-Specificity | 0.9510 |
| Macro-Precision | 0.9068 |
| AUC-PR (Macro) | 0.9751 |
| Cohen's Kappa | 0.8536 |
| MCC | 0.8565 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 4179.0 (0.24 ms/张) |
| End-to-End FPS | 190.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9038 | 0.9484 | 0.9082 | 0.9060 | 0.9458 | 0.9797 | 0.9668 | 208 |
| Malignant_lesions | 0.7955 | 0.9775 | 0.9396 | 0.8615 | 0.9157 | 0.9824 | 0.9651 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9271 | 0.8727 | 0.9320 | 1.0000 | 0.9968 | 0.9933 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=19 TN=349 FN=20
- **Malignant_lesions**: TP=140 FP=9 TN=391 FN=36
- **body_of_stomach_normal**: TP=192 FP=28 TN=356 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 9 | 11 |
| Malignant_lesions | 19 | 140 | 17 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9082    0.9038    0.9060       208
     Malignant_lesions     0.9396    0.7955    0.8615       176
body_of_stomach_normal     0.8727    1.0000    0.9320       192

              accuracy                         0.9028       576
             macro avg     0.9068    0.8998    0.8999       576
          weighted avg     0.9060    0.9028    0.9011       576

```
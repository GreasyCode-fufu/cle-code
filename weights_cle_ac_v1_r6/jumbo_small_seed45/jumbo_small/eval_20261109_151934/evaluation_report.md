# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/jumbo_small_seed45/jumbo_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9199 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9917 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9214 |
| Balanced Acc | 0.9180 |
| Macro-Sensitivity | 0.9180 |
| Macro-Specificity | 0.9608 |
| Macro-Precision | 0.9325 |
| AUC-PR (Macro) | 0.9850 |
| Cohen's Kappa | 0.8846 |
| MCC | 0.8890 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 694.7 (1.44 ms/张) |
| End-to-End FPS | 195.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9212 | 0.8755 | 0.9252 | 0.9883 | 0.9883 | 0.9812 | 208 |
| Malignant_lesions | 0.7784 | 0.9950 | 0.9856 | 0.8698 | 0.9108 | 0.9889 | 0.9774 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9661 | 0.9363 | 0.9646 | 0.9973 | 0.9979 | 0.9963 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=29 TN=339 FN=4
- **Malignant_lesions**: TP=137 FP=2 TN=398 FN=39
- **body_of_stomach_normal**: TP=191 FP=13 TN=371 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 2 | 2 |
| Malignant_lesions | 28 | 137 | 11 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8755    0.9808    0.9252       208
     Malignant_lesions     0.9856    0.7784    0.8698       176
body_of_stomach_normal     0.9363    0.9948    0.9646       192

              accuracy                         0.9236       576
             macro avg     0.9325    0.9180    0.9199       576
          weighted avg     0.9294    0.9236    0.9214       576

```
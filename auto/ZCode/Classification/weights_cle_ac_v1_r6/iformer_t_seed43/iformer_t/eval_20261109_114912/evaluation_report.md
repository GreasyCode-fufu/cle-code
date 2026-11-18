# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed43/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9367 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9370 |
| Balanced Acc | 0.9353 |
| Macro-Sensitivity | 0.9353 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9402 |
| AUC-PR (Macro) | 0.9825 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9068 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 5841.1 (0.17 ms/张) |
| End-to-End FPS | 228.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9867 | 0.9744 | 208 |
| Malignant_lesions | 0.8693 | 0.9850 | 0.9623 | 0.9134 | 0.9448 | 0.9852 | 0.9757 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=153 FP=6 TN=394 FN=23
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 6 | 5 |
| Malignant_lesions | 19 | 153 | 4 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9623    0.8693    0.9134       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9375       576
             macro avg     0.9402    0.9353    0.9367       576
          weighted avg     0.9386    0.9375    0.9370       576

```
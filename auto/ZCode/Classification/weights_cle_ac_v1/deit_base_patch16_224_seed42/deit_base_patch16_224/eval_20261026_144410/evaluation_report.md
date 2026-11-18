# deit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_base_patch16_224_seed42/deit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9777 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9814 |
| Balanced Acc | 0.9838 |
| Macro-Sensitivity | 0.9838 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9723 |
| AUC-PR (Macro) | 0.9862 |
| Cohen's Kappa | 0.9704 |
| MCC | 0.9706 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 346.3 (2.89 ms/张) |
| End-to-End FPS | 158.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9957 | 0.9947 | 0.9792 | 0.9706 | 0.9922 | 0.9944 | 195 |
| Malignant_lesions | 0.9873 | 0.9828 | 0.9286 | 0.9571 | 0.9971 | 0.9939 | 0.9641 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=1 TN=231 FN=7
- **Malignant_lesions**: TP=78 FP=6 TN=342 FN=1
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 6 | 1 |
| Malignant_lesions | 1 | 78 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9947    0.9641    0.9792       195
     Malignant_lesions     0.9286    0.9873    0.9571        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9723    0.9838    0.9777       427
          weighted avg     0.9820    0.9813    0.9814       427

```
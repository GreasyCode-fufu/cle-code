# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7822 |
| Accuracy | 0.7882 |
| Macro-AUROC (OvR) | 0.9272 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7834 |
| Balanced Acc | 0.7845 |
| Macro-Sensitivity | 0.7845 |
| Macro-Specificity | 0.8931 |
| Macro-Precision | 0.7954 |
| AUC-PR (Macro) | 0.8691 |
| Cohen's Kappa | 0.6806 |
| MCC | 0.6865 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 1782.7 (0.56 ms/张) |
| End-to-End FPS | 251.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7692 | 0.8723 | 0.7729 | 0.7711 | 0.8699 | 0.8951 | 0.8458 | 208 |
| Malignant_lesions | 0.6364 | 0.9475 | 0.8421 | 0.7249 | 0.8555 | 0.9211 | 0.8227 | 176 |
| body_of_stomach_normal | 0.9479 | 0.8594 | 0.7712 | 0.8505 | 0.9706 | 0.9653 | 0.9387 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=160 FP=47 TN=321 FN=48
- **Malignant_lesions**: TP=112 FP=21 TN=379 FN=64
- **body_of_stomach_normal**: TP=182 FP=54 TN=330 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 160 | 17 | 31 |
| Malignant_lesions | 41 | 112 | 23 |
| body_of_stomach_normal | 6 | 4 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7729    0.7692    0.7711       208
     Malignant_lesions     0.8421    0.6364    0.7249       176
body_of_stomach_normal     0.7712    0.9479    0.8505       192

              accuracy                         0.7882       576
             macro avg     0.7954    0.7845    0.7822       576
          weighted avg     0.7935    0.7882    0.7834       576

```
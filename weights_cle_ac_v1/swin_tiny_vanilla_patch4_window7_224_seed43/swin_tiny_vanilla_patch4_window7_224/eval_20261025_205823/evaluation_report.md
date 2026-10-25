# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9274 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9903 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9275 |
| Balanced Acc | 0.9249 |
| Macro-Sensitivity | 0.9249 |
| Macro-Specificity | 0.9628 |
| Macro-Precision | 0.9322 |
| AUC-PR (Macro) | 0.9832 |
| Cohen's Kappa | 0.8901 |
| MCC | 0.8914 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 474.2 (2.11 ms/张) |
| End-to-End FPS | 185.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9185 | 0.8684 | 0.9083 | 0.9713 | 0.9886 | 0.9816 | 208 |
| Malignant_lesions | 0.8750 | 0.9750 | 0.9390 | 0.9059 | 0.9466 | 0.9838 | 0.9704 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9948 | 0.9891 | 0.9681 | 0.9745 | 0.9985 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=30 TN=338 FN=10
- **Malignant_lesions**: TP=154 FP=10 TN=390 FN=22
- **body_of_stomach_normal**: TP=182 FP=2 TN=382 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 21 | 154 | 1 |
| body_of_stomach_normal | 9 | 1 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8684    0.9519    0.9083       208
     Malignant_lesions     0.9390    0.8750    0.9059       176
body_of_stomach_normal     0.9891    0.9479    0.9681       192

              accuracy                         0.9271       576
             macro avg     0.9322    0.9249    0.9274       576
          weighted avg     0.9302    0.9271    0.9275       576

```
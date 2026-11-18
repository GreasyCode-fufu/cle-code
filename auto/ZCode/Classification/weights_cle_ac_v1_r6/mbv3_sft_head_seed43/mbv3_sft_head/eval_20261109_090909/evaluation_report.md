# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed43/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9266 |
| Accuracy | 0.9415 |
| Macro-AUROC (OvR) | 0.9903 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9411 |
| Balanced Acc | 0.9271 |
| Macro-Sensitivity | 0.9271 |
| Macro-Specificity | 0.9707 |
| Macro-Precision | 0.9264 |
| AUC-PR (Macro) | 0.9660 |
| Cohen's Kappa | 0.9070 |
| MCC | 0.9072 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2428.4 (0.41 ms/张) |
| End-to-End FPS | 145.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9333 | 0.9655 | 0.9579 | 0.9455 | 0.9451 | 0.9894 | 0.9887 | 195 |
| Malignant_lesions | 0.8481 | 0.9684 | 0.8590 | 0.8535 | 0.9656 | 0.9819 | 0.9097 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9781 | 0.9623 | 0.9808 | 1.0000 | 0.9997 | 0.9995 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=182 FP=8 TN=224 FN=13
- **Malignant_lesions**: TP=67 FP=11 TN=337 FN=12
- **body_of_stomach_normal**: TP=153 FP=6 TN=268 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 182 | 11 | 2 |
| Malignant_lesions | 8 | 67 | 4 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9579    0.9333    0.9455       195
     Malignant_lesions     0.8590    0.8481    0.8535        79
body_of_stomach_normal     0.9623    1.0000    0.9808       153

              accuracy                         0.9415       427
             macro avg     0.9264    0.9271    0.9266       427
          weighted avg     0.9412    0.9415    0.9411       427

```
# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed46/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9556 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9645 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9804 |
| Macro-Precision | 0.9636 |
| AUC-PR (Macro) | 0.9893 |
| Cohen's Kappa | 0.9438 |
| MCC | 0.9442 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3008.2 (0.33 ms/张) |
| End-to-End FPS | 92.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9526 | 0.9455 | 0.9622 | 0.9822 | 0.9958 | 0.9949 | 195 |
| Malignant_lesions | 0.8734 | 0.9885 | 0.9452 | 0.9079 | 0.9718 | 0.9927 | 0.9730 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=11 TN=221 FN=4
- **Malignant_lesions**: TP=69 FP=4 TN=344 FN=10
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 10 | 69 | 0 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9455    0.9795    0.9622       195
     Malignant_lesions     0.9452    0.8734    0.9079        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9649       427
             macro avg     0.9636    0.9488    0.9556       427
          weighted avg     0.9650    0.9649    0.9645       427

```
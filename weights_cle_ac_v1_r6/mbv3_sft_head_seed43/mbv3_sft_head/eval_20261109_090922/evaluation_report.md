# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed43/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8285 |
| Accuracy | 0.8351 |
| Macro-AUROC (OvR) | 0.9717 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8301 |
| Balanced Acc | 0.8287 |
| Macro-Sensitivity | 0.8287 |
| Macro-Specificity | 0.9158 |
| Macro-Precision | 0.8524 |
| AUC-PR (Macro) | 0.9493 |
| Cohen's Kappa | 0.7506 |
| MCC | 0.7588 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3316.6 (0.30 ms/张) |
| End-to-End FPS | 105.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8702 | 0.8587 | 0.7768 | 0.8209 | 0.9213 | 0.9526 | 0.9264 | 208 |
| Malignant_lesions | 0.6420 | 0.9825 | 0.9417 | 0.7635 | 0.8618 | 0.9724 | 0.9399 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9062 | 0.8386 | 0.9012 | 0.9858 | 0.9901 | 0.9814 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=181 FP=52 TN=316 FN=27
- **Malignant_lesions**: TP=113 FP=7 TN=393 FN=63
- **body_of_stomach_normal**: TP=187 FP=36 TN=348 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 181 | 6 | 21 |
| Malignant_lesions | 48 | 113 | 15 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7768    0.8702    0.8209       208
     Malignant_lesions     0.9417    0.6420    0.7635       176
body_of_stomach_normal     0.8386    0.9740    0.9012       192

              accuracy                         0.8351       576
             macro avg     0.8524    0.8287    0.8285       576
          weighted avg     0.8478    0.8351    0.8301       576

```
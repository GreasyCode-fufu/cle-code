# mbv3_r6_s2_lsside 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsside_seed42/mbv3_r6_s2_lsside/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9408 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9899 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9409 |
| Balanced Acc | 0.9394 |
| Macro-Sensitivity | 0.9394 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9431 |
| AUC-PR (Macro) | 0.9794 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9116 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 4341.5 (0.23 ms/张) |
| End-to-End FPS | 239.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9851 | 0.9732 | 208 |
| Malignant_lesions | 0.8920 | 0.9800 | 0.9515 | 0.9208 | 0.9538 | 0.9851 | 0.9662 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=157 FP=8 TN=392 FN=19
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 17 | 157 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9515    0.8920    0.9208       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9410       576
             macro avg     0.9431    0.9394    0.9408       576
          weighted avg     0.9418    0.9410    0.9409       576

```
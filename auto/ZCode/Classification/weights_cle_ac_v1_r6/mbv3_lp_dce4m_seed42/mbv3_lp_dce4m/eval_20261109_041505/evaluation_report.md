# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lp_dce4m_seed42/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9409 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9409 |
| Balanced Acc | 0.9398 |
| Macro-Sensitivity | 0.9398 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9432 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9116 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2737.0 (0.37 ms/张) |
| End-to-End FPS | 15.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9538 | 0.9206 | 0.9336 | 0.9696 | 0.9927 | 0.9887 | 208 |
| Malignant_lesions | 0.9034 | 0.9875 | 0.9695 | 0.9353 | 0.9587 | 0.9968 | 0.9935 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9688 | 0.9394 | 0.9538 | 0.9841 | 0.9960 | 0.9918 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=17 TN=351 FN=11
- **Malignant_lesions**: TP=159 FP=5 TN=395 FN=17
- **body_of_stomach_normal**: TP=186 FP=12 TN=372 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 5 | 6 |
| Malignant_lesions | 11 | 159 | 6 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9206    0.9471    0.9336       208
     Malignant_lesions     0.9695    0.9034    0.9353       176
body_of_stomach_normal     0.9394    0.9688    0.9538       192

              accuracy                         0.9410       576
             macro avg     0.9432    0.9398    0.9409       576
          weighted avg     0.9418    0.9410    0.9409       576

```
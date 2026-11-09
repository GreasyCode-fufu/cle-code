# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed44/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9616 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9617 |
| Balanced Acc | 0.9600 |
| Macro-Sensitivity | 0.9600 |
| Macro-Specificity | 0.9804 |
| Macro-Precision | 0.9648 |
| AUC-PR (Macro) | 0.9953 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9431 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4400.4 (0.23 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9565 | 0.9273 | 0.9533 | 0.9888 | 0.9953 | 0.9920 | 208 |
| Malignant_lesions | 0.9148 | 0.9950 | 0.9877 | 0.9499 | 0.9637 | 0.9976 | 0.9949 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=16 TN=352 FN=4
- **Malignant_lesions**: TP=161 FP=2 TN=398 FN=15
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 2 | 2 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9273    0.9808    0.9533       208
     Malignant_lesions     0.9877    0.9148    0.9499       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9618       576
             macro avg     0.9648    0.9600    0.9616       576
          weighted avg     0.9631    0.9618    0.9617       576

```
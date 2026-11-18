# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed45/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9652 |
| Balanced Acc | 0.9618 |
| Macro-Sensitivity | 0.9618 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9511 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9444 |
| MCC | 0.9447 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4088.5 (0.24 ms/张) |
| End-to-End FPS | 219.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9487 | 0.9828 | 0.9788 | 0.9635 | 0.9580 | 0.9964 | 0.9958 | 195 |
| Malignant_lesions | 0.9367 | 0.9713 | 0.8810 | 0.9080 | 0.9854 | 0.9937 | 0.9742 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=185 FP=4 TN=228 FN=10
- **Malignant_lesions**: TP=74 FP=10 TN=338 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 185 | 10 | 0 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9788    0.9487    0.9635       195
     Malignant_lesions     0.8810    0.9367    0.9080        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9649       427
             macro avg     0.9511    0.9618    0.9561       427
          weighted avg     0.9660    0.9649    0.9652       427

```
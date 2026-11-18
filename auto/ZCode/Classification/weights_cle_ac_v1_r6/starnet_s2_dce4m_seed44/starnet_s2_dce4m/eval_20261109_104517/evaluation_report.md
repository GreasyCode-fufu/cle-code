# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed44/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9629 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9693 |
| Balanced Acc | 0.9602 |
| Macro-Sensitivity | 0.9602 |
| Macro-Specificity | 0.9839 |
| Macro-Precision | 0.9658 |
| AUC-PR (Macro) | 0.9881 |
| Cohen's Kappa | 0.9515 |
| MCC | 0.9516 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3491.2 (0.29 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9741 | 0.9692 | 0.9692 | 0.9741 | 0.9961 | 0.9955 | 195 |
| Malignant_lesions | 0.9114 | 0.9885 | 0.9474 | 0.9290 | 0.9801 | 0.9933 | 0.9688 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=6 TN=226 FN=6
- **Malignant_lesions**: TP=72 FP=4 TN=344 FN=7
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 4 | 2 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9692    0.9692    0.9692       195
     Malignant_lesions     0.9474    0.9114    0.9290        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9696       427
             macro avg     0.9658    0.9602    0.9629       427
          weighted avg     0.9693    0.9696    0.9693       427

```
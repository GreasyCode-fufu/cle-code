# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed44/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9612 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9693 |
| Balanced Acc | 0.9572 |
| Macro-Sensitivity | 0.9572 |
| Macro-Specificity | 0.9839 |
| Macro-Precision | 0.9656 |
| AUC-PR (Macro) | 0.9862 |
| Cohen's Kappa | 0.9514 |
| MCC | 0.9515 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3542.4 (0.28 ms/张) |
| End-to-End FPS | 7.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9741 | 0.9695 | 0.9745 | 0.9826 | 0.9950 | 0.9939 | 195 |
| Malignant_lesions | 0.8987 | 0.9885 | 0.9467 | 0.9221 | 0.9773 | 0.9902 | 0.9650 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9891 | 0.9806 | 0.9870 | 0.9963 | 0.9999 | 0.9998 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=6 TN=226 FN=4
- **Malignant_lesions**: TP=71 FP=4 TN=344 FN=8
- **body_of_stomach_normal**: TP=152 FP=3 TN=271 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 5 | 71 | 3 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9695    0.9795    0.9745       195
     Malignant_lesions     0.9467    0.8987    0.9221        79
body_of_stomach_normal     0.9806    0.9935    0.9870       153

              accuracy                         0.9696       427
             macro avg     0.9656    0.9572    0.9612       427
          weighted avg     0.9693    0.9696    0.9693       427

```
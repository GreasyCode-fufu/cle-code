# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed45/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9425 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9425 |
| Balanced Acc | 0.9422 |
| Macro-Sensitivity | 0.9422 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9430 |
| AUC-PR (Macro) | 0.9828 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4567.3 (0.22 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9565 | 0.9234 | 0.9257 | 0.9591 | 0.9853 | 0.9779 | 208 |
| Malignant_lesions | 0.9091 | 0.9750 | 0.9412 | 0.9249 | 0.9606 | 0.9884 | 0.9720 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=16 TN=352 FN=15
- **Malignant_lesions**: TP=160 FP=10 TN=390 FN=16
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 10 | 5 |
| Malignant_lesions | 14 | 160 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9234    0.9279    0.9257       208
     Malignant_lesions     0.9412    0.9091    0.9249       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9427       576
             macro avg     0.9430    0.9422    0.9425       576
          weighted avg     0.9425    0.9427    0.9425       576

```
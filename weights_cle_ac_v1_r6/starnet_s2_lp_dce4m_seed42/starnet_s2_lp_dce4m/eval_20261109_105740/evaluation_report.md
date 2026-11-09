# starnet_s2_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_lp_dce4m_seed42/starnet_s2_lp_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9287 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9286 |
| Balanced Acc | 0.9281 |
| Macro-Sensitivity | 0.9281 |
| Macro-Specificity | 0.9640 |
| Macro-Precision | 0.9304 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.8929 |
| MCC | 0.8933 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5240.5 (0.19 ms/张) |
| End-to-End FPS | 7.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9511 | 0.9139 | 0.9161 | 0.9537 | 0.9859 | 0.9789 | 208 |
| Malignant_lesions | 0.8920 | 0.9800 | 0.9515 | 0.9208 | 0.9538 | 0.9935 | 0.9863 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9609 | 0.9257 | 0.9492 | 0.9866 | 0.9954 | 0.9907 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=18 TN=350 FN=17
- **Malignant_lesions**: TP=157 FP=8 TN=392 FN=19
- **body_of_stomach_normal**: TP=187 FP=15 TN=369 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 8 | 9 |
| Malignant_lesions | 13 | 157 | 6 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9139    0.9183    0.9161       208
     Malignant_lesions     0.9515    0.8920    0.9208       176
body_of_stomach_normal     0.9257    0.9740    0.9492       192

              accuracy                         0.9288       576
             macro avg     0.9304    0.9281    0.9287       576
          weighted avg     0.9293    0.9288    0.9286       576

```
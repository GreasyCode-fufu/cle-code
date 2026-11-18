# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed44/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9730 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9990 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9789 |
| Balanced Acc | 0.9721 |
| Macro-Sensitivity | 0.9721 |
| Macro-Specificity | 0.9892 |
| Macro-Precision | 0.9739 |
| AUC-PR (Macro) | 0.9964 |
| Cohen's Kappa | 0.9665 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1444.4 (0.69 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9828 | 0.9795 | 0.9795 | 0.9828 | 0.9991 | 0.9990 | 195 |
| Malignant_lesions | 0.9367 | 0.9885 | 0.9487 | 0.9427 | 0.9857 | 0.9979 | 0.9903 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=4 TN=228 FN=4
- **Malignant_lesions**: TP=74 FP=4 TN=344 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9795    0.9795    0.9795       195
     Malignant_lesions     0.9487    0.9367    0.9427        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9789       427
             macro avg     0.9739    0.9721    0.9730       427
          weighted avg     0.9788    0.9789    0.9789       427

```
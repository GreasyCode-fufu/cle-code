# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9565 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9565 |
| Balanced Acc | 0.9581 |
| Macro-Sensitivity | 0.9581 |
| Macro-Specificity | 0.9786 |
| Macro-Precision | 0.9560 |
| AUC-PR (Macro) | 0.9937 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9354 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 669.7 (1.49 ms/张) |
| End-to-End FPS | 210.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9864 | 0.9745 | 0.9455 | 0.9553 | 0.9951 | 0.9917 | 208 |
| Malignant_lesions | 0.9716 | 0.9625 | 0.9194 | 0.9448 | 0.9872 | 0.9960 | 0.9915 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=5 TN=363 FN=17
- **Malignant_lesions**: TP=171 FP=15 TN=385 FN=5
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 14 | 3 |
| Malignant_lesions | 3 | 171 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9745    0.9183    0.9455       208
     Malignant_lesions     0.9194    0.9716    0.9448       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9566       576
             macro avg     0.9560    0.9581    0.9565       576
          weighted avg     0.9576    0.9566    0.9565       576

```
# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet34_seed44/resnet34/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9247 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9857 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9252 |
| Balanced Acc | 0.9223 |
| Macro-Sensitivity | 0.9223 |
| Macro-Specificity | 0.9619 |
| Macro-Precision | 0.9303 |
| AUC-PR (Macro) | 0.9735 |
| Cohen's Kappa | 0.8874 |
| MCC | 0.8890 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 2242.7 (0.45 ms/张) |
| End-to-End FPS | 251.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9185 | 0.8690 | 0.9108 | 0.9741 | 0.9811 | 0.9672 | 208 |
| Malignant_lesions | 0.8466 | 0.9775 | 0.9430 | 0.8922 | 0.9354 | 0.9777 | 0.9571 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9982 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=30 TN=338 FN=9
- **Malignant_lesions**: TP=149 FP=9 TN=391 FN=27
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 26 | 149 | 1 |
| body_of_stomach_normal | 4 | 3 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8690    0.9567    0.9108       208
     Malignant_lesions     0.9430    0.8466    0.8922       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9253       576
             macro avg     0.9303    0.9223    0.9247       576
          weighted avg     0.9282    0.9253    0.9252       576

```
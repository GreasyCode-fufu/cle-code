# convnext_wds_wtb_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_msf_seed42/convnext_wds_wtb_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9564 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9556 |
| Macro-Sensitivity | 0.9556 |
| Macro-Specificity | 0.9780 |
| Macro-Precision | 0.9575 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9348 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.56G |
| FLOPs | 9.11G |
| Params | 29.55M |
| Pure-Inference FPS | 1145.3 (0.87 ms/张) |
| End-to-End FPS | 253.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9620 | 0.9346 | 0.9479 | 0.9779 | 0.9929 | 0.9893 | 208 |
| Malignant_lesions | 0.9261 | 0.9825 | 0.9588 | 0.9422 | 0.9680 | 0.9929 | 0.9833 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=14 TN=354 FN=8
- **Malignant_lesions**: TP=163 FP=7 TN=393 FN=13
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9346    0.9615    0.9479       208
     Malignant_lesions     0.9588    0.9261    0.9422       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9566       576
             macro avg     0.9575    0.9556    0.9564       576
          weighted avg     0.9568    0.9566    0.9566       576

```
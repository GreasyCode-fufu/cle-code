# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9522 |
| Balanced Acc | 0.9494 |
| Macro-Sensitivity | 0.9494 |
| Macro-Specificity | 0.9760 |
| Macro-Precision | 0.9575 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9293 |
| MCC | 0.9312 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1068.0 (0.94 ms/张) |
| End-to-End FPS | 231.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9904 | 0.9538 | 0.9238 | 0.9559 | 0.9943 | 0.9953 | 0.9915 | 208 |
| Malignant_lesions | 0.8580 | 0.9975 | 0.9934 | 0.9207 | 0.9410 | 0.9941 | 0.9873 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9766 | 0.9552 | 0.9771 | 1.0000 | 0.9999 | 0.9998 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=206 FP=17 TN=351 FN=2
- **Malignant_lesions**: TP=151 FP=1 TN=399 FN=25
- **body_of_stomach_normal**: TP=192 FP=9 TN=375 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 206 | 1 | 1 |
| Malignant_lesions | 17 | 151 | 8 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9238    0.9904    0.9559       208
     Malignant_lesions     0.9934    0.8580    0.9207       176
body_of_stomach_normal     0.9552    1.0000    0.9771       192

              accuracy                         0.9531       576
             macro avg     0.9575    0.9494    0.9512       576
          weighted avg     0.9555    0.9531    0.9522       576

```
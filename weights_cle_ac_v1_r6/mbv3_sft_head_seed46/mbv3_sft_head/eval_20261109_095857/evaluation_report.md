# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed46/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8619 |
| Accuracy | 0.8628 |
| Macro-AUROC (OvR) | 0.9768 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8625 |
| Balanced Acc | 0.8573 |
| Macro-Sensitivity | 0.8573 |
| Macro-Specificity | 0.9295 |
| Macro-Precision | 0.8800 |
| AUC-PR (Macro) | 0.9579 |
| Cohen's Kappa | 0.7926 |
| MCC | 0.7989 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 5572.8 (0.18 ms/张) |
| End-to-End FPS | 181.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.8424 | 0.7698 | 0.8435 | 0.9568 | 0.9647 | 0.9437 | 208 |
| Malignant_lesions | 0.7330 | 0.9775 | 0.9348 | 0.8217 | 0.8927 | 0.9750 | 0.9470 | 176 |
| body_of_stomach_normal | 0.9062 | 0.9688 | 0.9355 | 0.9206 | 0.9538 | 0.9907 | 0.9829 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=58 TN=310 FN=14
- **Malignant_lesions**: TP=129 FP=9 TN=391 FN=47
- **body_of_stomach_normal**: TP=174 FP=12 TN=372 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 8 | 6 |
| Malignant_lesions | 41 | 129 | 6 |
| body_of_stomach_normal | 17 | 1 | 174 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7698    0.9327    0.8435       208
     Malignant_lesions     0.9348    0.7330    0.8217       176
body_of_stomach_normal     0.9355    0.9062    0.9206       192

              accuracy                         0.8628       576
             macro avg     0.8800    0.8573    0.8619       576
          weighted avg     0.8755    0.8628    0.8625       576

```
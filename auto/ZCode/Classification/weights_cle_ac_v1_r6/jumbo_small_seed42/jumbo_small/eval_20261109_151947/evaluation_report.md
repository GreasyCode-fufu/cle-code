# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/jumbo_small_seed42/jumbo_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9427 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9430 |
| Balanced Acc | 0.9418 |
| Macro-Sensitivity | 0.9418 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9440 |
| AUC-PR (Macro) | 0.9795 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9140 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 885.3 (1.13 ms/张) |
| End-to-End FPS | 216.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9847 | 0.9742 | 208 |
| Malignant_lesions | 0.9148 | 0.9675 | 0.9253 | 0.9200 | 0.9627 | 0.9835 | 0.9664 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9974 | 0.9946 | 0.9788 | 0.9821 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=161 FP=13 TN=387 FN=15
- **body_of_stomach_normal**: TP=185 FP=1 TN=383 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 15 | 161 | 0 |
| body_of_stomach_normal | 4 | 3 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9253    0.9148    0.9200       176
body_of_stomach_normal     0.9946    0.9635    0.9788       192

              accuracy                         0.9427       576
             macro avg     0.9440    0.9418    0.9427       576
          weighted avg     0.9436    0.9427    0.9430       576

```
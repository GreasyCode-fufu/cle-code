# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed42/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9725 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9788 |
| Balanced Acc | 0.9696 |
| Macro-Sensitivity | 0.9696 |
| Macro-Specificity | 0.9889 |
| Macro-Precision | 0.9757 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1915.8 (0.52 ms/张) |
| End-to-End FPS | 36.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9828 | 0.9796 | 0.9821 | 0.9870 | 0.9978 | 0.9974 | 195 |
| Malignant_lesions | 0.9241 | 0.9914 | 0.9605 | 0.9419 | 0.9829 | 0.9943 | 0.9765 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=4 TN=228 FN=3
- **Malignant_lesions**: TP=73 FP=3 TN=345 FN=6
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 4 | 73 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9796    0.9846    0.9821       195
     Malignant_lesions     0.9605    0.9241    0.9419        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9789       427
             macro avg     0.9757    0.9696    0.9725       427
          weighted avg     0.9788    0.9789    0.9788       427

```
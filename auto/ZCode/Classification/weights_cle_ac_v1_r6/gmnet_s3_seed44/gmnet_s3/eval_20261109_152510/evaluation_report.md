# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed44/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9190 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9196 |
| Balanced Acc | 0.9159 |
| Macro-Sensitivity | 0.9159 |
| Macro-Specificity | 0.9589 |
| Macro-Precision | 0.9289 |
| AUC-PR (Macro) | 0.9851 |
| Cohen's Kappa | 0.8794 |
| MCC | 0.8826 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 955.9 (1.05 ms/张) |
| End-to-End FPS | 214.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9049 | 0.8517 | 0.9054 | 0.9794 | 0.9896 | 0.9827 | 208 |
| Malignant_lesions | 0.8125 | 0.9875 | 0.9662 | 0.8827 | 0.9229 | 0.9883 | 0.9759 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9844 | 0.9688 | 0.9688 | 0.9844 | 0.9981 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=35 TN=333 FN=7
- **Malignant_lesions**: TP=143 FP=5 TN=395 FN=33
- **body_of_stomach_normal**: TP=186 FP=6 TN=378 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 29 | 143 | 4 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8517    0.9663    0.9054       208
     Malignant_lesions     0.9662    0.8125    0.8827       176
body_of_stomach_normal     0.9688    0.9688    0.9688       192

              accuracy                         0.9201       576
             macro avg     0.9289    0.9159    0.9190       576
          weighted avg     0.9257    0.9201    0.9196       576

```
# efficientvim_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/efficientvim_m1_seed43/efficientvim_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9613 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9691 |
| Balanced Acc | 0.9527 |
| Macro-Sensitivity | 0.9527 |
| Macro-Specificity | 0.9825 |
| Macro-Precision | 0.9719 |
| AUC-PR (Macro) | 0.9877 |
| Cohen's Kappa | 0.9512 |
| MCC | 0.9517 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.46M |
| FLOPs | 468.92M |
| Params | 5.71M |
| Pure-Inference FPS | 4308.0 (0.23 ms/张) |
| End-to-End FPS | 245.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9569 | 0.9505 | 0.9673 | 0.9867 | 0.9954 | 0.9944 | 195 |
| Malignant_lesions | 0.8734 | 0.9943 | 0.9718 | 0.9200 | 0.9719 | 0.9926 | 0.9688 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=10 TN=222 FN=3
- **Malignant_lesions**: TP=69 FP=2 TN=346 FN=10
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 10 | 69 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9505    0.9846    0.9673       195
     Malignant_lesions     0.9718    0.8734    0.9200        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9696       427
             macro avg     0.9719    0.9527    0.9613       427
          weighted avg     0.9699    0.9696    0.9691       427

```
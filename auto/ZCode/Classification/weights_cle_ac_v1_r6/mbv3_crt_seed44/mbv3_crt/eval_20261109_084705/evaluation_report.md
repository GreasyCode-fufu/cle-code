# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed44/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9444 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9926 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9445 |
| Balanced Acc | 0.9424 |
| Macro-Sensitivity | 0.9424 |
| Macro-Specificity | 0.9716 |
| Macro-Precision | 0.9482 |
| AUC-PR (Macro) | 0.9861 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9172 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1711.1 (0.58 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9375 | 0.8973 | 0.9306 | 0.9801 | 0.9895 | 0.9825 | 208 |
| Malignant_lesions | 0.8920 | 0.9850 | 0.9632 | 0.9263 | 0.9540 | 0.9915 | 0.9810 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9969 | 0.9950 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=23 TN=345 FN=7
- **Malignant_lesions**: TP=157 FP=6 TN=394 FN=19
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 17 | 157 | 2 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8973    0.9663    0.9306       208
     Malignant_lesions     0.9632    0.8920    0.9263       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9444       576
             macro avg     0.9482    0.9424    0.9444       576
          weighted avg     0.9464    0.9444    0.9445       576

```
# repvit_m1 评估报告

- 权重: `weights_3class_full_seed44/repvit_m1/repvit_m1_3class_full_seed44_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7816 |
| Accuracy | 0.7794 |
| Macro-AUROC (OvR) | 0.9112 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7812 |
| Balanced Acc | 0.7762 |
| Macro-Sensitivity | 0.7762 |
| Macro-Specificity | 0.8873 |
| Macro-Precision | 0.7925 |
| AUC-PR (Macro) | 0.8323 |
| Cohen's Kappa | 0.6658 |
| MCC | 0.6678 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3867.6 (0.26 ms/张) |
| End-to-End FPS | 205.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8242 | 0.8265 | 0.7456 | 0.7829 | 0.8840 | 0.8959 | 0.8336 | 256 |
| Malignant_lesions | 0.7238 | 0.8633 | 0.7070 | 0.7153 | 0.8728 | 0.8697 | 0.7297 | 210 |
| body_of_stomach_normal | 0.7805 | 0.9721 | 0.9249 | 0.8466 | 0.9096 | 0.9681 | 0.9336 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=211 FP=72 TN=343 FN=45
- **Malignant_lesions**: TP=152 FP=63 TN=398 FN=58
- **body_of_stomach_normal**: TP=160 FP=13 TN=453 FN=45

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 211 | 37 | 8 |
| Malignant_lesions | 53 | 152 | 5 |
| body_of_stomach_normal | 19 | 26 | 160 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7456    0.8242    0.7829       256
     Malignant_lesions     0.7070    0.7238    0.7153       210
body_of_stomach_normal     0.9249    0.7805    0.8466       205

              accuracy                         0.7794       671
             macro avg     0.7925    0.7762    0.7816       671
          weighted avg     0.7883    0.7794    0.7812       671

```
# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9038 |
| Accuracy | 0.9204 |
| Macro-AUROC (OvR) | 0.9777 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9198 |
| Balanced Acc | 0.9033 |
| Macro-Sensitivity | 0.9033 |
| Macro-Specificity | 0.9589 |
| Macro-Precision | 0.9046 |
| AUC-PR (Macro) | 0.9452 |
| Cohen's Kappa | 0.8734 |
| MCC | 0.8735 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 1735.8 (0.58 ms/张) |
| End-to-End FPS | 240.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9128 | 0.9397 | 0.9271 | 0.9199 | 0.9277 | 0.9763 | 0.9742 | 195 |
| Malignant_lesions | 0.8101 | 0.9626 | 0.8312 | 0.8205 | 0.9571 | 0.9594 | 0.8659 | 79 |
| body_of_stomach_normal | 0.9869 | 0.9745 | 0.9557 | 0.9711 | 0.9926 | 0.9973 | 0.9953 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=178 FP=14 TN=218 FN=17
- **Malignant_lesions**: TP=64 FP=13 TN=335 FN=15
- **body_of_stomach_normal**: TP=151 FP=7 TN=267 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 178 | 13 | 4 |
| Malignant_lesions | 12 | 64 | 3 |
| body_of_stomach_normal | 2 | 0 | 151 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9271    0.9128    0.9199       195
     Malignant_lesions     0.8312    0.8101    0.8205        79
body_of_stomach_normal     0.9557    0.9869    0.9711       153

              accuracy                         0.9204       427
             macro avg     0.9046    0.9033    0.9038       427
          weighted avg     0.9196    0.9204    0.9198       427

```
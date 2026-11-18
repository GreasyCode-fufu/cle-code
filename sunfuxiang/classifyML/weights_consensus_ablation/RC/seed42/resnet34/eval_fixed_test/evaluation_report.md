# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/RC/seed42/resnet34/resnet34_RC_seed42_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8999 |
| Accuracy | 0.8987 |
| Macro-AUROC (OvR) | 0.9813 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8976 |
| Balanced Acc | 0.9056 |
| Macro-Sensitivity | 0.9056 |
| Macro-Specificity | 0.9499 |
| Macro-Precision | 0.8993 |
| AUC-PR (Macro) | 0.9667 |
| Cohen's Kappa | 0.8480 |
| MCC | 0.8507 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3784.7 (0.26 ms/张) |
| End-to-End FPS | 242.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8125 | 0.9687 | 0.9412 | 0.8721 | 0.8933 | 0.9727 | 0.9630 | 256 |
| Malignant_lesions | 0.9190 | 0.9219 | 0.8428 | 0.8793 | 0.9615 | 0.9743 | 0.9440 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9592 | 0.9140 | 0.9484 | 0.9933 | 0.9970 | 0.9931 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=208 FP=13 TN=402 FN=48
- **Malignant_lesions**: TP=193 FP=36 TN=425 FN=17
- **body_of_stomach_normal**: TP=202 FP=19 TN=447 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 208 | 33 | 15 |
| Malignant_lesions | 13 | 193 | 4 |
| body_of_stomach_normal | 0 | 3 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9412    0.8125    0.8721       256
     Malignant_lesions     0.8428    0.9190    0.8793       210
body_of_stomach_normal     0.9140    0.9854    0.9484       205

              accuracy                         0.8987       671
             macro avg     0.8993    0.9056    0.8999       671
          weighted avg     0.9021    0.8987    0.8976       671

```
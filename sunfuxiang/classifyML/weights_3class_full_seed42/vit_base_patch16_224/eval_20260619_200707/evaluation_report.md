# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/vit_base_patch16_224/vit_base_patch16_224_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8647 |
| Accuracy | 0.8629 |
| Macro-AUROC (OvR) | 0.9575 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8621 |
| Balanced Acc | 0.8694 |
| Macro-Sensitivity | 0.8694 |
| Macro-Specificity | 0.9317 |
| Macro-Precision | 0.8638 |
| AUC-PR (Macro) | 0.9270 |
| Cohen's Kappa | 0.7943 |
| MCC | 0.7962 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 869.4 (1.15 ms/张) |
| End-to-End FPS | 209.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7852 | 0.9422 | 0.8933 | 0.8358 | 0.8767 | 0.9426 | 0.9212 | 256 |
| Malignant_lesions | 0.8524 | 0.8937 | 0.7851 | 0.8174 | 0.9300 | 0.9406 | 0.8873 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9592 | 0.9128 | 0.9409 | 0.9868 | 0.9894 | 0.9723 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=24 TN=391 FN=55
- **Malignant_lesions**: TP=179 FP=49 TN=412 FN=31
- **body_of_stomach_normal**: TP=199 FP=19 TN=447 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 43 | 12 |
| Malignant_lesions | 24 | 179 | 7 |
| body_of_stomach_normal | 0 | 6 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8933    0.7852    0.8358       256
     Malignant_lesions     0.7851    0.8524    0.8174       210
body_of_stomach_normal     0.9128    0.9707    0.9409       205

              accuracy                         0.8629       671
             macro avg     0.8638    0.8694    0.8647       671
          weighted avg     0.8654    0.8629    0.8621       671

```
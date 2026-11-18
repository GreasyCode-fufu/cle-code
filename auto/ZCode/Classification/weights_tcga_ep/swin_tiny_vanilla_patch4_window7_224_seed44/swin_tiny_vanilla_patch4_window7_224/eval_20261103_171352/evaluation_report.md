# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7892 |
| Accuracy | 0.8274 |
| Macro-AUROC (OvR) | 0.9618 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8247 |
| Balanced Acc | 0.7925 |
| Macro-Sensitivity | 0.7925 |
| Macro-Specificity | 0.9643 |
| Macro-Precision | 0.7925 |
| AUC-PR (Macro) | 0.8645 |
| Cohen's Kappa | 0.7777 |
| MCC | 0.7787 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1623.8 (0.62 ms/张) |
| End-to-End FPS | 466.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9045 | 0.9950 | 0.7759 | 0.8353 | 0.9982 | 0.9926 | 0.9120 | 199 |
| cancer | 0.8983 | 0.9483 | 0.8519 | 0.8745 | 0.9657 | 0.9726 | 0.9383 | 2645 |
| empty | 0.9886 | 0.9642 | 0.9242 | 0.9553 | 0.9948 | 0.9943 | 0.9871 | 3256 |
| normal | 0.7389 | 0.9858 | 0.9105 | 0.8158 | 0.9509 | 0.9606 | 0.8958 | 1735 |
| other | 0.6570 | 0.9248 | 0.6270 | 0.6416 | 0.9333 | 0.9302 | 0.7548 | 1717 |
| stroma | 0.5675 | 0.9677 | 0.6652 | 0.6125 | 0.9518 | 0.9207 | 0.6989 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=180 FP=52 TN=10383 FN=19
- **cancer**: TP=2376 FP=413 TN=7576 FN=269
- **empty**: TP=3219 FP=264 TN=7114 FN=37
- **normal**: TP=1282 FP=126 TN=8773 FN=453
- **other**: TP=1128 FP=671 TN=8246 FN=589
- **stroma**: TP=614 FP=309 TN=9243 FN=468

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 180 | 4 | 1 | 4 | 9 | 1 |
| cancer | 9 | 2376 | 0 | 39 | 139 | 82 |
| empty | 0 | 0 | 3219 | 0 | 37 | 0 |
| normal | 1 | 151 | 1 | 1282 | 181 | 119 |
| other | 40 | 161 | 262 | 19 | 1128 | 107 |
| stroma | 2 | 97 | 0 | 64 | 305 | 614 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7759    0.9045    0.8353       199
      cancer     0.8519    0.8983    0.8745      2645
       empty     0.9242    0.9886    0.9553      3256
      normal     0.9105    0.7389    0.8158      1735
       other     0.6270    0.6570    0.6416      1717
      stroma     0.6652    0.5675    0.6125      1082

    accuracy                         0.8274     10634
   macro avg     0.7925    0.7925    0.7892     10634
weighted avg     0.8269    0.8274    0.8247     10634

```
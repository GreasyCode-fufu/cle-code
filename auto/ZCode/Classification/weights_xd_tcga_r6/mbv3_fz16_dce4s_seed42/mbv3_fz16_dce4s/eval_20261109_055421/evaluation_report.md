# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8357 |
| Accuracy | 0.8693 |
| Macro-AUROC (OvR) | 0.9771 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8710 |
| Balanced Acc | 0.8402 |
| Macro-Sensitivity | 0.8402 |
| Macro-Specificity | 0.9739 |
| Macro-Precision | 0.8348 |
| AUC-PR (Macro) | 0.8894 |
| Cohen's Kappa | 0.8350 |
| MCC | 0.8357 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9127.7 (0.11 ms/张) |
| End-to-End FPS | 541.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8100 | 0.9952 | 0.7500 | 0.7788 | 0.9966 | 0.9936 | 0.8415 | 100 |
| cancer | 0.8901 | 0.9563 | 0.8610 | 0.8753 | 0.9663 | 0.9809 | 0.9512 | 1329 |
| empty | 0.9932 | 0.9975 | 0.9938 | 0.9935 | 0.9973 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8314 | 0.9923 | 0.9590 | 0.8906 | 0.9645 | 0.9907 | 0.9691 | 1014 |
| other | 0.7418 | 0.9627 | 0.7837 | 0.7621 | 0.9534 | 0.9445 | 0.8568 | 879 |
| stroma | 0.7747 | 0.9391 | 0.6614 | 0.7136 | 0.9645 | 0.9533 | 0.7179 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=81 FP=27 TN=5575 FN=19
- **cancer**: TP=1183 FP=191 TN=4182 FN=146
- **empty**: TP=1610 FP=10 TN=4071 FN=11
- **normal**: TP=843 FP=36 TN=4652 FN=171
- **other**: TP=652 FP=180 TN=4643 FN=227
- **stroma**: TP=588 FP=301 TN=4642 FN=171

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 81 | 2 | 2 | 2 | 6 | 7 |
| cancer | 20 | 1183 | 0 | 16 | 41 | 69 |
| empty | 0 | 0 | 1610 | 0 | 9 | 2 |
| normal | 4 | 54 | 0 | 843 | 42 | 71 |
| other | 3 | 58 | 8 | 6 | 652 | 152 |
| stroma | 0 | 77 | 0 | 12 | 82 | 588 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7500    0.8100    0.7788       100
      cancer     0.8610    0.8901    0.8753      1329
       empty     0.9938    0.9932    0.9935      1621
      normal     0.9590    0.8314    0.8906      1014
       other     0.7837    0.7418    0.7621       879
      stroma     0.6614    0.7747    0.7136       759

    accuracy                         0.8693      5702
   macro avg     0.8348    0.8402    0.8357      5702
weighted avg     0.8758    0.8693    0.8710      5702

```
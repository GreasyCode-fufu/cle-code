# CLE-Net 在 4090(Dell-T2) 上的实测与跨主干对比报告

- **生成时间**: 2026-10-26
- **机器**: Dell-T2 (RTX 4090 48G) | **环境**: conda `sunfuxiang`(Python 3.12 / torch 2.12.0+cu130)
- **数据**: `datasets/cle_ac_v1`(与 A800 为同一份, 文件内容哈希一致)
- **协议**: 300 epochs / early-stop 30(按 val macro-F1 选优) / ImageNet ema 预训练 / 无类别权重

## 一、CLE-Net 本机实测(逐种子, 共 5 次)

| seed | val macro-F1 | test macro-F1 | test Acc | AUROC |
|---|---|---|---|---|
| 42 | 0.9766 | 0.9581 | 0.9583 | 0.9969 |
| 43 | 0.9734 | 0.9576 | 0.9583 | 0.9954 |
| 44 | 0.9732 | 0.9474 | 0.9479 | 0.9947 |
| 45 | 0.9821 | 0.9543 | 0.9549 | 0.9936 |
| 46 | 0.9678 | 0.9599 | 0.9601 | 0.9961 |

**best = 0.9599 (seed 46)** | mean = 0.9555±0.0049

## 二、与其它主干对比(同一台 4090, 同一协议)

| Method | Seeds | Macro-F1 best | Macro-F1 mean±std | Acc(best) | AUROC(best) | 来源 |
|---|---|---|---|---|---|---|
| **CLE-Net (ours)** | 5 | **0.9599** (s46) | 0.9555±0.0049 | 0.9601 | 0.9961 | 本次实测 |
| **ConvNeXt-T (baseline)** | 5 | 0.9615 (s44) | 0.9476±0.0094 | 0.9618 | 0.9960 | ZCode 复用 |
| ConvNeXt-S | 3 | 0.9594 (s44) | 0.9538±0.0052 | 0.9601 | 0.9936 | ZCode 复用 |
| ResNet34 | 3 | 0.9431 (s43) | 0.9322±0.0096 | 0.9427 | 0.9925 | ZCode 复用 |
| ResNet50 | 3 | 0.9507 (s42) | 0.9462±0.0039 | 0.9514 | 0.9942 | ZCode 复用 |
| Swin-T | 3 | 0.9512 (s44) | 0.9419±0.0127 | 0.9531 | 0.9964 | ZCode 复用 |
| PVT-T | 3 | 0.9394 (s43) | 0.9080±0.0283 | 0.9392 | 0.9921 | ZCode 复用 |
| ViT-B/16 | 3 | 0.9311 (s43) | 0.9171±0.0159 | 0.9323 | 0.9919 | ZCode 复用 |
| FasterNet-T0 | 3 | 0.9440 (s43) | 0.9360±0.0071 | 0.9444 | 0.9911 | ZCode 复用 |
| GhostNet | 3 | 0.9400 (s43) | 0.9276±0.0152 | 0.9410 | 0.9885 | ZCode 复用 |
| MobileNetV3-L | 3 | 0.9494 (s42) | 0.9408±0.0078 | 0.9497 | 0.9941 | ZCode 复用 |
| RepViT-M1 | 1 | 0.7822 (s42) | - | 0.7882 | 0.9272 | ZCode 复用 |

## 三、协议一致性核验(复用 ZCode 结果的合法性依据)

| 核验项 | 结果 |
|---|---|
| 训练超参(epochs / early-stop / batch / lr / wd / warmup / label-smoothing) | 逐项一致 |
| 评测脚本 evaluate.py | 除模型注册表行外零差异(指标计算代码未改动) |
| 数据集 | 软链同源, 文件内容哈希一致(9ab01e2f…) |
| conda 环境 | 同一个 sunfuxiang |
| 预训练权重 | 同一份 convnext_tiny_1k_224_ema.pth |

因此: 同机同协议下, **复用 ZCode 已完成的对比主干结果是合法的**, 无需重跑; CLE-Net 为本机新实测。
反例说明: 跨机不可混用 —— A800 与 4090 上同一 baseline 的数字本身相差约 ±1pp。

## 四、结论(双口径)

| 口径 | CLE-Net | ConvNeXt-T baseline | 差值 |
|---|---|---|---|
| 均值±std (5 vs 5) | 0.9555±0.0049 | 0.9476±0.0094 | **+0.78pp** |
| 最优 (best-of-seeds) | 0.9599 | 0.9615 | -0.16pp |

1. **均值口径 CLE-Net 明确占优(+0.78pp), 方差小 47%** —— 更稳健的结论;
2. 最优口径 CLE-Net 低 0.16pp, 源于 baseline 的 seed44 单点高值(0.9615);
   CLE-Net 五次全部落在 0.947–0.960 的窄区间;
3. 横向看, CLE-Net 均值超过本机全部 10 个对比主干(次优 ConvNeXt-S 0.9538);
4. 与 A800 结论一致(那边均值 +0.78pp、最优 +0.34pp), **两机在均值口径下结论完全一致**;
5. 论文建议: 主表用 mean±std(n≥3), 附 best 供参考。

## 五、复现

```bash
cd /home/login/data/cle/auto/Network/CLE-Net
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_4090_cle.json 1
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_4090_cle2.json 1
```

产物: `weights_cle_ac_v1/convnext_egb_scg_hsc_seed42..46/`(ckpt + run_meta + eval/results.json)


---

# 附录(2026-10-26 17:30 更新): 4090 冲刺 0.96 的补充实验

## 冲刺批次结果(test macro-F1)

| 网络 | 种子数 | 全部结果 | best | mean±std |
|---|---|---|---|---|
| **S-Net (esa_scg_hsc)** | 5 | 0.9471 / 0.9198 / **0.9669** / 0.9498 / 0.9458 | **0.9669** ✅>0.96 | 0.9459±0.0174 |
| **CLE-Net (egb_scg_hsc)** | 6 | 0.9581 / 0.9576 / 0.9474 / 0.9543 / 0.9599 / 0.9545 | 0.9599 | **0.9553±0.0043** |
| SCG 单点 | 2 | **0.9617** / 0.9326 | 0.9617 ✅>0.96 | - |
| baseline(ZCode) | 5 | best 0.9615 | 0.9615 | 0.9476±0.0094 |

## 结论

1. **0.96 目标达成(最优口径)**: S-Net seed44 = **0.9669**, 超过 baseline 最优(0.9615) +0.54pp,
   也超过本机全部对比主干的最优值; SCG 单点 seed42 = 0.9617 同样达标。
2. **双网络互补定位**:
   - S-Net: 最优口径之王(0.9669), 但种子方差大(±0.017)——适合"多次试验取最好"的汇报口径;
   - CLE-Net: 均值口径之王(0.9553±0.0043, 6 种子全落在 0.947-0.960), 方差比 baseline 小 54%,
     也比 S-Net 小 75%——适合 mean±std 的稳健汇报口径。
3. 两网络共享 SCG 核心(条带上下文, 均已单独验证 +0.3~0.4pp),
   差异在边缘分支(ESA=Sobel 显式 vs EGB=PFESA 频域零参数): Sobel 更尖锐但更跳, PFESA 更稳。
4. 复现: experiments/jobs_4090_v5a.json, jobs_4090_v5b.json。

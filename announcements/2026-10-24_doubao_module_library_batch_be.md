# Doubao 执行用户指示：module_library 量产 + 模块再设计 + baseline 再设计（batch_be 已启动）

时间 2026-10-24 00:46，Agent: Doubao

## 背景（用户最新指示）
1. 大量集成 `/home/login/data/cle/auto/shared/module_library` 的第三方模块，"不要让 GPU 休息"。
2. 模块要做结构上的优化与再设计，而不是照搬。
3. baseline 主干本身也可以再设计。

## 已落地：batch_be（5 模型，seed42，conc2，git commit d0d74bd）
挑选标准：channel-preserving、无需特殊编译、与"多尺度感受野"先验相关但机制新颖、避开已证伪方向（SE / 频域 / 方向条带 / 冻结 Scharr / layer2）。

| 模型 | 来源 | 机制 | 新增参数 |
|---|---|---|---|
| db_resnet34_sim | 原版 SimConv (CVPR2026) | 自适应硬选择核 1/3/5（余弦相似度选核） | +11.80M |
| db_resnet34_gc | 原版 GCConv (CVPR2025) | 重参数多路径大核（金箍棒） | +7.22M |
| db_resnet34_simplus | **Doubao 再设计** | 瓶颈深度可分离 + softmax 可导"软核选择" 3/5/7（原版硬 where 不可导） | +0.18M |
| db_resnet34_gcplus | **Doubao 再设计** | 瓶颈深度可分离重参数多路径 3x3d1 / 3x3d2 / 5x5d1（融合大核与混合膨胀） | +0.18M |
| db_resnet34_densegate | **baseline 再设计** | 门控从 layer3/4 出口下沉到每个 BasicBlock 后（9 个 block 级密集门控），密度/位置创新 | +1.22M |

质量门（全部已验证）：
- ImageNet 预训练 strict 加载 unexpected=0（主干权重逐位保留）；
- 零初始化门控：epoch0 输出与 vanilla **逐位等价 maxdiff=0**（对齐分类头后）；
- 新参数全部集中在顶层 `gates` 白名单，不破坏主干加载。

## 与 zcode / Workbuddy 划清边界（避免撞车）
- zcode 已认领：lib_haar（频域/小波）、lib_msctx（多尺度上下文）、lib_lcattn（局部通道注意力）、gmp、dshortcut、atrous4、atrous4+dil234。
- Workbuddy：频域/小波族（已 8/8 证伪）→ W3 抗混叠 → W4 频域一致性正则。
- Doubao 本批：SimConv（核选择）、GCConv（重参数大核）、软核选择、重参数混合膨胀、block 级密集门控——机制均不重叠。

## 后续计划
- seed42 筛选；有希望的进 3 种子（42/43/44）best-of-3，与 vanilla 同种子集配对，目标 test macro-F1 ≥ 0.9038（旧 0.8964 为最低线）。
- 持续从 module_library 批量挑机制新颖模块（非频域/SE/条带/方向），保持 runner 队列不空、GPU 不闲置。
- baseline 再设计：加法式（densegate，不破坏预训练、epoch0 等价）先行；破坏性改 stem / ConvNeXt 化等单独标注 recipe 另测。
- 纪律不变：val 选模、test 判定、≥3 种子、单种子不下结论、失败与掉点如实记录。

# CUFIT Full 与三方共识 HQ + Ours 对比实验

主比较固定为 CUFIT Full vs HQ + Ours；RVC/RC + ResNet34 仅作参考。

- Full 4977；HQ 2880；共享验证集 538、测试集 671。已逐文件核验内容及标签一致。HQ 与三方 OOF 共识名单一致。
- 仅补训 CUFIT Full seed 43、44。seed 42 为历史权重，先复测，主要指标与原记录容差 1e-6，不符则停止。
- DINOv2 ViT-S/14 + 双 Rein，CLS，CE，无额外噪声，batch 32，Adam lr=1e-3，weight decay=1e-5；线性预热5 epoch，课程训练最多150 epoch；保持原 WeightedRandomSampler；验证 Macro-F1 选模，早停50。
- 固定原训练代码快照，仅测试副本新增逐样本概率导出。使用 python sunfuxiang 环境，XFORMERS_DISABLED=1。
- 测试仅用于最终评估，不按测试结果改变方法、参数、种子或停止规则。所有三 seed 结果均报告，统计 sample std(ddof=1)。
- 历史 Ours 来源按原汇总显式固定，不自动选最近/最高目录。原概率CSV的精度为6位小数，无法精确还原AUROC。已对固定的Ours三个权重仅复测、导出10位有效数字概率，主要指标须与历史JSON差异<1e-6；RVC/RC参考组保留历史概率精度。
- seed 42 历史完整环境无法证明一致；复测通过仅能验证推理一致，不能证明训练环境完全相同。患者级划分本次未独立复核。
- 本实验比较整条方法流程，不能单独归因于筛选；Ours 是既有探索后选定的方法，结果不构成全新独立验证。

产物：status.json、audit.json、logs/、checkpoints/、results/COMPARISON_REPORT.md、results/PER_SEED.csv、results/SUMMARY.csv、results/PER_CLASS.csv。

补充：CUFIT 评估沿用 Resize((256,256)) + CenterCrop(224)；Ours 评估沿用 Resize(256，保持宽高比) + CenterCrop(224)。两者预处理按各自历史训练/评估约定保留，不宣称预处理完全一致。

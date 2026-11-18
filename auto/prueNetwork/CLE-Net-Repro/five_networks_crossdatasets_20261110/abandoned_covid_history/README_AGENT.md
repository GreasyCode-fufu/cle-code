# 五网络跨数据集实验 Agent入口
本实验由Codex独立运行：MambaOut-Tiny、TransNeXt-Tiny、MambaVision-T、ThinkingViT、CPUBone-B2。
Covid、BUSI、CUB各自独立训练，种子42/43/44，共45次。不是CLE_ac训练后直接外测，也不是ZCode的Jumbo/GmNet/iFormer/LSNet批次。
不新增CLE-Net或baseline训练。停止的MobileNet-V3实验保持停止。
code/为独立代码快照；source_sha256.json与pretrained_sha256.json记录源码和初始化权重。环境版本见environment.json和pip_freeze.txt。
数据只读使用既有固定划分。protocol.json记录真实路径、类别名、样本数量和训练配置；dataset_manifest.json记录逐图像哈希。cross_split_duplicate_audit.json列出精确跨集合重复，不因检查结果悄悄更改划分；患者独立性未验证。
启动程序原件为/tmp/codex_run_five_crossdatasets.py，对应本目录run_experiment.py。训练和评估均经code/latest_top_networks/run_train.py、run_eval.py执行，注意评估class-names必须取对应数据集真实类别。
224输入、ImageNet预训练与归一化、AdamW lr1e-4 wd0.05、bs32、cosine300、early-stop30、unweighted CE、AMP。训练仅按val Macro-F1选checkpoint，完成后评估固定测试集。ThinkingViT继续使用既有3H→6H、threshold0、最终阶段CE适配。
status.json为当前状态；runs/包含每次训练日志、run_meta、最佳权重、TRAIN_DONE标记和eval_test结果。只有测试JSON成功生成才计为完成。
per_seed.csv和summary.csv/json会随着已完成运行增量更新。n<3的汇总只能作为中间结果，不得作为完整3种子表引用；样本标准差ddof=1。
恢复完整运行可跳过已完成测试，但有中断checkpoint且没有TRAIN_DONE时程序会主动失败，禁止未经检查将不完整训练当作正常结束或悄悄从头改实验。
GPU每次启动子任务前检查占用，有其他进程则等待，不终止其他Agent任务。未完成时不得写成45/45已完成。

## 已获用户明确选择：保留历史划分
字节哈希审计发现Covid跨集合重复104组，其中训练—测试重复55组。BUSI和CUB未发现字节完全相同的跨集合重复。用户明确选择沿用历史划分完成本批并标注限制；不得擅自去重重分，也不得将Covid结果表述为严格无泄漏的跨域泛化验证。最终Excel每个Covid结果表必须写明此限制。

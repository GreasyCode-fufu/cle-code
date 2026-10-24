# A–C第三轮：先检验更新食谱和主干表征

第二轮已全部独立审计：plain .973445、近似参数卷积 .971243、原LS .972929、正核 .972872、空间平均核 .969753。LS对plain平均-.052百分点，两种子负；四家族区间全跨零，共同主要门槛false。停止继续枚举LS核变体。十五模型末轮训练准确率99.9%–100%，val全部低于best，plain47最佳epoch2、最终由.9736降至.9481。这提示应检验预训练表征的更新幅度/主干选择；不证明标签噪声或过拟合的唯一原因。

仅固定A–C train2077/val427，原test576不加载。四流程×相同seed45/46/47：cnn_original复用第二轮3个已审计plain；新增cnn_low_lr、dino_frozen、dino_full各三，共9新神经训练、12比较模型。原AC清单/标签/增强/60epoch/batch32/AdamWwd.05/cosine0/动态AMP不变。新三流程直接调用不可改R10的transfer.train_job；CNN和完整DINO都encoder1e-5/head1e-4、unscale后clip1，冻结DINO仅768→3头lr1e-4。原CNN参照所有参数lr1e-4且无clip。低LR和clip共同变化，差值是更新食谱整体作用，不能单独归因LR。

cnn_low_lr仅把原ResNet34拆为encoder(含原avgpool)和head(原fc)，前向和初始参数逐位等价；全部21286211参数训练，普通BN仍更新。DINO使用已有官方DINOv2 ViT-S/14无register公开LVD142M权重，经未改pretrained_representations.py/timm1.0.27 port，原CLS+patch均值768线性头；两DINO初始encoder/head相同，冻结头2307参数，完整21631107。已有权重SHA b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9不重下。公开Apache2.0许可沿用；timm位置插值与官方offset差异已披露，不声称官方精确推理。预训练数据/架构/精度/计算/特征维数不同，不能归因单一自监督或结构创新。A–C按CNN一致性定义也可能偏向CNN，因此结果只描述该固定数据集。

四预定比较共用39患者×20000bootstrap RNG20261024、普通95与四家族[.625,99.375]。问题一：cnn_low_lr对cnn_original，三个seed差均正、平均>=+.5百分点、家族下界正才支持此更新食谱。问题二：dino_full对cnn_original、cnn_low_lr、dino_frozen三项都达同要求才支持该完整DINO系统。两个问题各自报告，不用某个成功单项替代失败完整条件。新判据不回改旧轮，不按中途结果改参考/seed/等级。val同时选best且反复使用，条件区间不覆盖所有选择/训练不确定性；不是新患者独立确认。

预检逐位验证CNN拆分前向/配对状态、两DINO初始head/source、精确可训练集/优化器唯一覆盖和LR、真实batch32有限AMP/clip更新、冻结DINO源参数不变、CNN BN计数和源权重变化、参数/FLOP/显存。所有新源/config/jobs/protocol和3参考SHA在正式训练前冻结提交。每模型保存60日志、最早最高val最佳ckpt、完整427概率、源状态哈希/漂移和训练预算。数据loader原范围不包含test，物理缓存仍含3595全split，不混淆。

资源初始6并发、自适应最多8，CPU线程1/loader2，每2秒记录GPU/CPU单核/I/O/RAM，0.5秒补队列。失败先查日志/活子进程，按签名/SHA复用有效训练；不干预其他Agent。全结果/3参照/独立审计/中文解释/预算/代码/配置/许可交付outputs/CLE_AC_dataset/backbone_round3，权重缓存留服务器。最终候选冻结后再统一评价固定ACtest576；本轮无test调参或新颖性承诺。

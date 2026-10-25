# CLE内容注意力对比：完整十五模型轮

15×60epoch全完成；427val/576test四角度两独立路线概率、实际权重/source/SHA、每epoch2077唯一索引及患者互斥审计通过。所有seed与阴性保留，本轮结束不等于整体完成。
条件数值门槛方案：无。仅为自适应内部条件数值结果，不代表新颖、独立泛化或外部验证。

|方法|s45 F1|s46 F1|s47 F1|均值|对B0差pp|条件CI pp|
|---|---:|---:|---:|---:|---:|---|
|mom_salience_single|94.73286%|94.60072%|94.60138%|94.64499%|-0.33645|[-1.27169,0.77877]|
|mom_salience_c4|96.50971%|95.47517%|94.92925%|95.63804%|0.65660|[-0.89465,2.04486]|
|salience_only_single|95.62175%|94.25282%|94.75396%|94.87617%|-0.10527|[-1.29902,1.00629]|
|salience_only_c4|95.97789%|95.81678%|95.64338%|95.81268%|0.83124|[-0.91705,2.03024]|
|mom_weightedmean_single|95.42269%|94.57926%|94.73786%|94.91327%|-0.06817|[-0.86012,0.85016]|
|mom_weightedmean_c4|96.65063%|95.12039%|94.74934%|95.50679%|0.52535|[-0.61824,1.63131]|
|mom_misaligned_single|96.01553%|95.63662%|94.51181%|95.38799%|0.40655|[-0.33500,1.49184]|
|mom_misaligned_c4|96.35941%|95.28514%|95.08609%|95.57688%|0.59544|[-0.38356,1.63929]|
|mom_ungated_single|95.45147%|95.28044%|94.40486%|95.04559%|0.06415|[-0.86256,0.98799]|
|mom_ungated_c4|95.63410%|95.26828%|95.63240%|95.51159%|0.53015|[-0.91341,1.64648]|

## 累计/去一/成熟控制

mom_salience加gated内容注意力对比；salience_only真去18528Mom；weightedmean不减GAP；misaligned改变权重和值配对；ungated移除U、参数更少非等容量。gated新26931、ungated14627，原GAP/learnednorm/head保留。分类器首步正；V/U/w首步0、第二步正，正式CUDA已核验。

|mom_salience比较|配对seed差pp|平均差pp|条件CI pp|
|---|---|---:|---|
|single_vs_M_moments|-0.91096,-1.38940,-1.38562|-1.22866|[-2.41055,-0.73470]|
|single_vs_salience_only|-0.88889,0.34790,-0.15258|-0.23119|[-0.97212,0.68616]|
|single_vs_mom_misaligned|-1.28267,-1.03591,0.08957|-0.74300|[-1.49878,-0.15564]|
|single_vs_mom_ungated|-0.71862,-0.67973,0.19652|-0.40061|[-1.07587,0.37167]|
|single_vs_mom_weightedmean|-0.68984,0.02146,-0.13647|-0.26829|[-1.04928,0.47041]|
|single_vs_B0|-0.51862,0.37489,-0.86563|-0.33645|[-1.27169,0.77877]|
|c4_vs_M_moments|-0.17889,-0.87099,-1.40977|-0.81988|[-2.49288,-0.07253]|
|c4_vs_salience_only|0.53182,-0.34161,-0.71413|-0.17464|[-0.87005,0.78038]|
|c4_vs_mom_misaligned|0.15030,0.19003,-0.15684|0.06116|[-1.01854,0.89424]|
|c4_vs_mom_ungated|0.87561,0.20689,-0.70315|0.12645|[-0.83884,1.00755]|
|c4_vs_mom_weightedmean|-0.14092,0.35478,0.17991|0.13125|[-1.23593,1.12861]|
|c4_vs_B0|1.25824,1.24934,-0.53777|0.65660|[-0.89465,2.04486]|

所有新arm对原Mom/成熟ECA/aug同视图比较见mechanism_comparisons.json，B0_C4/DINO_C4/Swin_C4等强参照完整保留于round_results。原Mom及固定B0有效三seed复用，不重训或削弱调参预算；成熟C4收益不能自动归因新结构。

|方案|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|
|---|---:|---:|---:|---:|---:|
|mom_salience_single|94.67593%|92.04545%|92.22226%|27867894|9.114330656|
|mom_salience_c4|95.65972%|93.75000%|92.85350%|27867894|36.457322624|
|salience_only_single|94.90741%|93.93939%|92.11529%|27849366|8.912021024|
|salience_only_c4|95.83333%|95.45455%|91.64832%|27849366|35.648084096|
|mom_weightedmean_single|94.96528%|91.09848%|92.42608%|27867894|9.114330656|
|mom_weightedmean_c4|95.54398%|92.04545%|92.66825%|27867894|36.457322624|
|mom_misaligned_single|95.42824%|92.61364%|92.71824%|27867894|9.114330656|
|mom_misaligned_c4|95.60185%|93.18182%|92.55994%|27867894|36.457322624|
|mom_ungated_single|95.08102%|91.85606%|92.81641%|27855590|9.113126432|
|mom_ungated_c4|95.54398%|92.42424%|93.00110%|27855590|36.452505728|

原Mom27840963/9.111840768G；gated候选27867894/9.114330656G，去Mom27849366/8.912021024G，ungated27855590/9.113126432G；B0为27822435/8.909531136G。C4四倍前向；逻辑2FLOP/MAC含新Linear和attention加权pooling BMM及原三额外旋转卷积，排除LN/softmax/tanh/sigmoid/中心化/逐元素/访存。成熟注意力/差值/组合不自动新颖，无临床定位或解释保证。实际吞吐另测。

## 实际资源

|真实父训练数|覆盖秒|时间加权GPU|GPU至少80%时长|CPU|峰显存MiB|
|---|---:|---:|---:|---:|---:|
|0|80.97|27.499%|19.999%|5.154%|1307|
|1|2200.22|87.935%|88.398%|8.390%|4256|

连续1127行覆盖2281.186秒，全部15次launch/PID/PPID采样核对；39行status滞后按真实父子进程恢复。低GPU区间保留，0父进程可能为管理器推理/CPU分析，不都为空闲。无记录不补造，不外推设计/CPU阶段满载，不虚称Agent连续在线。profile cap不是正式并发。

|arm|epoch图秒|加载等待比例|元数据队列图秒|
|---|---:|---:|---:|
|mom_salience|905.05|1.622%|823.88|
|salience_only|1094.93|2.013%|983.19|
|mom_weightedmean|904.86|1.602%|824.71|
|mom_misaligned|901.39|1.631%|821.06|
|mom_ungated|906.67|1.610%|825.87|

## 复现、使用与科学边界

固定首val2386实际checkpoint单图CLI对原batch32最大概率差2.5510787963867188e-05<预设1e-4，类别一致；非三seed效果或临床验证。原Q/M A224 recipe、结构、预测命令与实际吞吐见STRUCTURE_AND_USE_ZH.md；实际权重SHA/最早valbest见weights_and_tasks.json。

# 特征内容注意力对比：无CLE图像CPU原型

原CNNMom继续保留。前一协方差15模型全轮完成；完整候选对原Mom single/C4点差均负且条件CI跨0，对通道独立置换两mode增量CI也跨0，不确认新增组件。single对diag条件CI正但seed方向混合，diag只有16有效坐标，不能倒置为对原Mom增量。不泛化否定全部联合统计。

下一互补假设是末层49个特征位置的信息贡献并不相同。原Mom刻画stem方向响应分布，新的只读出末层内容选择。保持原GAP/learned norm/head；在768×7×7特征每位置做无仿射channel LN1e-5，gated score = w[tanh(Vh)×sigmoid(Uh)]，rank16，w无bias（常数bias会在softmax取消）。softmax在49位置归一。候选d=Σ a_i(f_i−mean(f))，768维无仿射LN1e-5与零768→3分类器加原logits。无CNN外部图像/患者信息、无区域标签/筛图/重采样。它只关注特征内容，不保证病灶定位或医学解释。

五arm：mom_salience累计；salience_only真去18528Mom；mom_weightedmean用相同gated score加权原f，不减均值；mom_misaligned对权重位置作固定default_rng(20261025)49置换再与未置换value相乘，保留权重和值的各自分布而改变配对；mom_ungated去sigmoid U，只保留成熟tanh attention。前三种gated新26931参数，ungated新14627参数，后者参数更少并非容量匹配，不隐藏差异。mom_only/plain仅初始化参照，不重训原有效Mom/B0参考。此前F的field_pool按原图支持掩码加权，本次是末层可学习内容权重，保留此前阴性，不混称新原理。

成熟来源：[Ilse等2018 PMLR论文](https://proceedings.mlr.press/v80/ilse18a.html)、[官方AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)。论文提出按实例内容学习归一权重及gated扩展。本地只适配已审阅pooling公式：加入score输入LN、去w无效bias、减GAP、descriptor LN及保留原分类路径零残差；不是原始MNIST-BAGS/病理MIL整网复现，不继承原文分类或解释性结论，不把注意力/差值/LN/残差或Mom组合称创新。

官方model.py与MIT LICENSE原字节保留，Gitblob分别e99adc74d63240c637f3a1ec3174c4a2d9f1aa26、58ed84464a359db312c3da36f040bfeba8f8572d，作者2018 Maximilian Ilse/Jakub Tomczak归属保留。仅受限AST提取GatedAttention.forward的A_V/A_U/A/Z六赋值做输出、输入及全部score参数梯度oracle，不执行模型训练入口。原型算子按公式本地实现。

所有CLE训练/val/test图0。CPU做独立NumPy线性/门控/softmax/加权和/LN、doublegradcheck、常数/退化矩形有限、联合位置排列不变（错配控制除外）、批独立、同GAP不同内容反例、两合成SGD更新及三seed×七完整公开224起点logit/RNG/原参数对象一致。零分类器首步使V/U/w梯度0；分类器首步正，第二步V/U/w正，正式CUDA也须真实batch32检查。softmax和LN会改变尺度，没有稳健性保证；联合位置不变不等于整个CNN等变，错配控制固定置换破坏此性质。

逻辑2FLOP/MAC包括新Linear、attention加权pooling BMM和原三额外旋转conv；不计LN/softmax/tanh/sigmoid/中心化/逐元素/访存，实际完整目标吞吐另测。通过CPU后才可新目录正式冻结5arm×3、原Q/M A224 CE/60epoch同recipe、六完整427val真实权重重建/CUDA32两更新/profile通过后开训。不能将CPU通过说成CLE增益。train→val最早最佳→test→下一设计；全部test自适应内部开发，条件CI不校正选择偏差。整体active，不保证提点、SCI新颖或录用。


## 正式15模型事前注册

5arm×45/46/47全部60epoch/batch32，原Q/M A224相同公开ImageNet和seed head、原CE；encoder1e-5，新模块/head1e-4，AdamW.05、cosine0/clip1/AMP、stochasticdepth.1，HV翻转/15度旋转/随机224裁剪、workers2/thread1；参考原Mom/B0有效三seed复用，不更换/削弱/少给调参预算。原型与原Mom逐字复制，库调用本原型完整MAC counter。

固定AC train2077/182患者、val427/39、test576/41、恶性1，患者可多标签；标签/等级/阈值/清单/患者互斥split/采样不改。物理3595缓存非训练3595；每epoch2077唯一索引SHA e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691。全部60epoch原0度val最早最大macroF1选权重，三seed完成登记实际checkpoint后再test四角度。独立直接A物理缓存固定索引与T TestData A路线完整427val/576test×4概率审计。

预注册mom_salience对原Mom/salience_only/mom_weightedmean/mom_misaligned/mom_ungated/B0两mode12直接整数Fraction控制；41患者20k条件CI，主比较保留原Mom/ECA/aug及同C4 B0/DINO/Swin。严格门槛为平均差>3/200、三seed全正、条件CI下界>0；自适应内部test，区间不校正选择偏差，非独立泛化/外部验证。去Mom支持不能倒置成新分支增量。

正式CPU、真实CUDA32两有限更新/全梯度有限、分类器首步正和V/U/w首步0第二步正/原Mom首步正/去Mom真移18528、原Mom三actual权重经mom_only严格load加B0共六完整427val概率严格0，全部缓存SHA/训练拒绝test/2077索引和每完整arm实测吞吐/6GiB余量/97%最小并发全过才开训。所有有效模型/阴性保留，普通阴性不停止整体。

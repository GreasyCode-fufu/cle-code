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

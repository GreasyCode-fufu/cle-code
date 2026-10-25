# 末层有界奇函数空间统计：无CLE图像原型

MERGE完整15模型在同Swin主干、同视图参照下的10组区间均跨0；局部分支对去local、去Mom、均值、全局控制也未形成确认增量。保留原CNN方向矩，不把Swin迁移或首merge局部分支带成已确认组件。新假设仅是：末层每通道空间分布的不对称性，可能补充原stem方向均值/标准差；这不是临床因果证据。

对ConvNeXt-Tiny末层B×768×7×7特征，逐图逐通道算μ、v；z=(x−μ)/sqrt(v+1e−6)。候选d=mean_spatial(tanh(z)^3)，再用无仿射channel LayerNorm(eps1e−5)，零768→3线性投影加到原分类logits。原avgpool→原learned norm→原head完整保留。新增2307参数；保留原Mom另18528参数。新模块构造用fork_rng，不热启动CLE权重。

互补性必须由相同μ/v、不同d的反例检查，不能由名称推定。d在LN前位于[-1,1]，LN后没有该界；它是有界奇函数统计，非标准Pearson偏度。空间排列不变，丢失位置；不识别病灶、不保证整网旋转等变。eps使正比例缩放只近似不变，标准化受离群点影响，不能据tanh称统计影响函数稳健。

五个候选arm：mom_odd保留Mom加有界奇统计；odd_only真去18528Mom；mom_raw3用相同z的三阶矩；mom_std用同一空间标准差；mom_even用相同tanh的偶二阶矩去符号。四种描述经同一无仿射LN和同样零2307参数投影。mom_only和plain只作原型初始化参照。未来若进入正式轮，保持原AC/CE/三seed/60epoch/val最早最佳/test自适应规则，并复用已有效原Mom和固定B0参照，做累计/去一/成熟矩控制；原型通过不自动授权声称效果。

成熟来源：[Spatial Moment Pooling Improves Neural Image Assessment](https://arxiv.org/abs/2209.14583)，作者论文PDF已读取，ICIP 2022。该论文研究均值/方差/三阶及更高阶空间矩，并在高阶矩后使用LayerNorm。本轮raw3仅为标准化三阶矩加同接口LN的受控适配；没有复制论文多阶拼接、IQA头或完整训练，不称SMP整法复现，也未查到可验证的作者实现。tanh、标准矩、残差、零初始化都不是独立创新证据。

本阶段只允许数学/NumPy/梯度/合成更新/公开ImageNet起点检查；CLE图、训练样本、val、test图全部0。CPU检查成功后还须另注册正式source/config、真实CUDA完整batch、父模型完整val重建、数据保护与吞吐profile，才能称已正式开训。整体研究active，阴性保留。

## 正式轮注册

本目录为正式十五模型：五arm×45/46/47，各60epoch/batch32/原AC训练2077唯一索引/原CE/AdamW权重衰减.05/encoder1e-5、新模块与head1e-4/cosine到0/clip1/AMP/HV翻转和15度旋转/随机224裁剪。原M与固定B0有效三seed复用，不削弱基线调参预算。每三seed全部val选最早最高权重后才读取test四角度，两独立路线全576概率重建。直接预注册mom_odd对原Mom/去Mom/raw3/std/even/B0两mode，整数Fraction及41患者20k条件CI。所有阴性保留、整体active。CPU/CUDA/六427概率重建/缓存白名单/profile未全通过不正式开训。

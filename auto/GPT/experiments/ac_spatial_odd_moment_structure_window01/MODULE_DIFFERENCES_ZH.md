# 来源和接口差异

spatial_odd_moment.py与c4_moment_stem.py逐字复制已完成CPU原型。原CNNMom source保持8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd。

原prototype对三seed×七变体公开初始logit/RNG/原参数ID严格一致、NumPy/反例/gradcheck/退化有限通过不替代本正式检查。新投影加logits、不替换原avgpool和learned norm，原Mom有效checkpoint以mom_only严格加载、完整427概率重建才可复用。去Mom只移除18528方向矩参数，新增2307投影相同。

[Spatial Moment Pooling作者论文](https://arxiv.org/abs/2209.14583)已经使用方差/偏度及高阶矩后LN。raw3为相同标准化z三阶矩与同LN的受控适配，不是作者多阶拼接/IQA整网复现；没有确认作者实现代码。不把成熟tanh/LN/零残差/组合称创新，不把统计空间排列不变推广到整网络等变。全部结果为test反馈的自适应内部开发，条件CI不校正选择偏差。

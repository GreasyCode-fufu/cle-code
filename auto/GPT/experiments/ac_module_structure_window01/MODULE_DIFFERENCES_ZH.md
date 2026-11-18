# 共享模块与官方实现差异审计

共享原件位于upstream/shared，官方抓取原件位于upstream/official；provenance.json记录官方Git blob SHA及URL，shared_snapshot_manifest.json记录共享原路径、字节SHA及login归属。不依据库文件名认定论文质量或创新。

1. **WTConv**：共享代码的2D多级分解、子带DW卷积、逆变换、base分支与[官方WTConv2d](https://github.com/BGU-CS-VIL/WTConv/blob/main/wtconv/wtconv2d.py)相同。共享版本以partial保存滤波器引用，stride>1使用固定1x1分组卷积；本轮stride1，官方池化差异不影响本轮。采用官方直接读取self滤波器的forward，固定滤波器从requires_grad=False Parameter改为buffer，防止训练参数统计混淆及partial设备引用问题。服务器无PyWavelets；只实现本轮注册的Haar/db1分析滤波器构造，拒绝其他wavelet。系数由sqrt(0.5)按官方float32顺序外积得到，记录固定buffer元素数。CPU核验共享/官方/本实现奇偶尺寸输出与输入梯度、独立NumPy DWT/IWT；此依赖适配不宣称新方法。原MIT许可证保留。CLE只在指定阶段末块插入，不冒充论文完整WTConvNeXt复现。[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf)
2. **SimAM**：共享PSAA中的SimAM仅返回sigmoid(energy)权重，虽名为SimAM也不能直接当官方特征模块使用；本实现明确返回x×gate，与[官方simam_module](https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py)在H×W>1且lambda1e-4时对齐。共享用max(HW−1,1)保护退化空间，本轮H14×W14，无差别。该模块不是“局部窗口”均值，而是各通道整幅特征图空间均值/方差；不沿用共享注释中的局部能量表述。[论文与作者页面](https://proceedings.mlr.press/v139/yang21o.html)
3. **ECA**：共享HCFNet ECA显式计算通道数到奇数kernel，C768时为5；[官方eca_layer](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py)默认k3、支持调用者显式指定k。对齐时传入k5并复制同一Conv1d权重，输出数值相同。reshape/transpose只是同一空间1×1通道序列接口差异，非创新；新增5个参数。MIT原许可保留。[官方仓库](https://github.com/BangguWu/ECANet)
4. **EMA**：共享默认factor8，[官方](https://github.com/YOLOonMe/EMA-attention-module/blob/main/EMA_attention_module)默认factor32并有未使用的c2参数。所有操作公式相同；本轮显式factor32，与官方逐元素/梯度对齐，不将分组差异当作结构收益。[作者论文](https://arxiv.org/abs/2305.13563)

官方SimAM与EMA根目录LICENSE请求未获得该文件，保留请求结果与来源，不能据此声称许可证已验证；本轮是用户授权的内部研究试验，未自动发布第三方代码或重新许可。

所有共享执行副本仅AST提取所需定义，添加明确依赖，未执行文件演示和其他未知模块代码；原件完整保留且共享库不写入。CPU/CUDA预检查失败则保留具体失败，再冻结修订版，不把未通过配置称为有效训练。

独立方向矩原型借鉴成熟旋转核lifting和group pooling，[Group Equivariant CNN](https://proceedings.mlr.press/v48/cohenc16.html)、[Cyclic Symmetry](https://proceedings.mlr.press/v48/dieleman16.html)作为已知方法背景。其均值/标准差与零投影预训练适配是否对CLE有用由本轮及后续重训消融判断；不预先宣称新颖性。

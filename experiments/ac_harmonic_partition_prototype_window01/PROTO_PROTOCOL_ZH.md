# C4方向功率形状：独立数学与公开初始化原型

保留原M方向矩工作信号，不因已有结构阴性否定所有设计。P患者对比完整组尚在运行，不改它的冻结方法/配置或据尚未完成控制挑方法；本原型只用合成张量、公开预训练权重及已有源文件，不读取CLE图像/预测，不进行CLE训练。正式新轮的机制取舍须等待P完整控制，不将本原型宣称研究提点。

动机来自已完成M/P细化：标准差比重复均值重要，但σ合并了全部方向变化能量。四个旋转滤波器响应可有相同µ/σ而有不同的相邻/相对方向关系。假设在CLE形态不依赖绝对摆放角度的情况下，保留这种方向周期结构可能补足σ；不把频谱形状直接等同腺体/核形态或医生标注正确性。

对r0,r1,r2,r3，a=(r0−r2)/2，b=(r1−r3)/2，c=(r0−r1+r2−r3)/4，E1=(a²+b²)/2，E2=c²，则人口方差var=E1+E2（C4 DFT Parseval）。提出有界功率对比q=(E2−E1)/(E1+E2+1e−6)，与保留µ/σ同时使用。q在C4循环移位下不变；全CNN普通层不受此约束，不能称整个分类器旋转等变/不变或连续360°等变。

在原Mom stem上保留原base和18528参数adapter对象，复用同四响应，只新增96→96零初始化1×1残差投影9312参数。原公开初始化logit、原参数及CPU/CUDA RNG应严格保留，原方向矩统计的数值/梯度不改。新描述不热启动已训练Mom模型；本原型只是公共网络有限反向，正式训练另行注册。

候选对照在此先明确算子语义，尚非新正式训练：shape=有界功率对比；permuted=同参数/边际响应但固定非循环排列[0,1,3,2]破坏方向邻接；duplicate_std=同新分支参数但重复已有σ，控制新增分支/优化重参数化；raw_h2=成熟DFT第二频率幅值sqrt(max(E2,1e−12))，控制归一化是否必要。所有保留原Mom，不无限枚举多个比例/eps。本原型可检验“q提供µ/σ没有的信息”、算子C4性与梯度，但不能验证CLE机制有效。

必须核：NumPy FFT独立频率功率/Parseval；同µσ不同q反例；四循环移位不变/非循环打乱破坏；输入方图90°旋转下局部descriptor空间协变；常量/单脉冲/低能量有限、正比例强度伸缩的eps偏差；double gradcheck；官方H-Net stack_magnitudes函数AST在受限NumPy/Torch shim执行，核raw_h2输出/输入梯度；公开ConvNeXt-Tiny224/Mom全logit、原参数ID、RNG、各分支梯度/预算。数值均在CPU，不创建CUDA训练上下文。

成熟先验必须披露：[Harmonic Networks CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Worrall_Harmonic_Networks_Deep_CVPR_2017_paper.html)使用圆谐波和复表示，[官方TensorFlow代码](https://github.com/danielewworrall/harmonicConvolutions)包含幅值算子；本方法只是C4离散响应后处理，不是H-Net整体复现。其官方stack_magnitudes为sqrt(max(sum_complex(x²),eps))，与本q不同；raw_h2严格复用幅值公式和默认eps=1e−12，保留MIT许可/原始blob/SHA。官方sum_magnitudes部分doc轴标签与README6D约定不一致，实际源码以complex轴4/order轴3计算，不能盲照文字猜轴。研究只执行白名单纯算子AST，不安装旧TensorFlow改变环境。

[DFT transformation invariant pooling ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Jongbin_Ryu_DFT-based_Transformation_Invariant_ECCV_2018_paper.html)、[显微图像旋转网络论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612823/)也已有DFT幅值/旋转先验。当前检索摘要可确认先验存在，但其全文获取返回403/验证码，不能声称已逐式全文复现。共享FMA静态源码是空间H×W的rfft2/irfft2及实虚部卷积，与此C4方向轴1D DFT不同，未将其代称本方法或FMA复现。共享库始终只读。

成熟频谱、群池化或这些操作组合不能自动称SCI创新。P及旧test均已有适应性反馈依赖，本原型以后也不能称全新独立泛化；最终固定B0、A–C清单、全部三seed、严格平均>0.015和患者条件区间门槛不变。不保证提升/新颖/录用。

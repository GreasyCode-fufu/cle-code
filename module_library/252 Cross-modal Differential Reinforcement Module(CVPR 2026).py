import torch
import torch.nn as nn

"""
    论文地址：hhttps://arxiv.org/pdf/2503.23359
    论文题目：VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion(CVPR 2026)
    中文题目：VideoFusion：面向多模态视频融合的时空协同网络 (CVPR 2026)
    讲解视频：https://www.bilibili.com/video/BV1ACWc6KEMi/
    跨模态差异增强模块（Cross-modal Differential Reinforcement Module，CmDRM）
        实际意义：①不同模态之间存在互补信息，但传统融合无法充分利用：红外与可见光特征既有对场景互相补充的信息，也存在大量重复冗余内容，直接拼接/相加特征会混入冗余。②单模态特征表征能力不足：只依靠单一模态自身内部特征，难以捕获另一模态的关键线索。③如何自适应决定不同模态的贡献比例：无法区分原始特征、差分增强特征对场景描述贡献。
        核心思想：通过挖掘模态间差异信息并进行自适应增强，充分利用不同模态的互补特征，提升多模态融合表示能力。
        代码流程：输入双模态特征 → 差异特征计算 → Cross Attention提取互补信息 → 自适应权重融合 → CA+SA增强关键区域 → 输出跨模态增强特征。
"""

# ========================= 第一部分：通道注意力（Channel Attention, CA） =========================
# 目的：学习每个通道的重要程度，突出有用的特征通道、抑制不重要的通道。
class ChannelAttention(nn.Module):
    # in_planes：输入通道数 C；ratio：中间隐藏通道的压缩比例，默认 16。
    def __init__(self, in_planes, ratio=16):
        # 初始化父类 nn.Module，注册后续定义的网络层。
        super(ChannelAttention, self).__init__()
        # 对每个通道做全局平均池化：[B, C, H, W] → [B, C, 1, 1]。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 对每个通道做全局最大池化：[B, C, H, W] → [B, C, 1, 1]。
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 第一个 1×1 卷积：通道数由 C 压缩到 C//ratio；用于学习通道间关系。
        # 注意：输入通道数需保证 in_planes // ratio >= 1，才能合法构建此层。
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # LeakyReLU 激活：负数部分仍保留 0.2 倍斜率，增强非线性表达。
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 第二个 1×1 卷积：将压缩后的通道恢复到原始 C 个通道。
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # Sigmoid：把通道响应映射到 (0, 1)，作为通道注意力权重。
        self.sigmoid = nn.Sigmoid()

    # 前向传播：输入 x 的维度为 [B, C, H, W]。
    def forward(self, x):
        # 平均池化分支：全局平均池化 → 降维卷积 → 激活 → 升维卷积。
        # 输出 avg_out 的维度为 [B, C, 1, 1]。
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        # 最大池化分支：全局最大池化 → 降维卷积 → 激活 → 升维卷积。
        # 输出 max_out 的维度同样为 [B, C, 1, 1]。
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        # 将平均池化和最大池化分支逐元素相加，融合两种通道统计信息。
        out = avg_out + max_out
        # 通过 Sigmoid 获得每个通道对应的权重，维度为 [B, C, 1, 1]。
        out = self.sigmoid(out)
        # 返回通道权重（此处尚未与原始输入 x 相乘）。
        return out


# ========================= 第二部分：空间注意力（Spatial Attention, SA） =========================
# 目的：学习不同空间位置的重要程度，重点关注缺陷、边缘等关键区域。
class SpatialAttention(nn.Module):
    # 初始化空间注意力模块；无须预先指定输入通道数。
    def __init__(self):
        # 初始化父类 nn.Module。
        super(SpatialAttention, self).__init__()
        # 3×3 卷积：把两张空间描述图 [B, 2, H, W] 融合成一张 [B, 1, H, W]。
        # padding=1 表示在四周填充 1 个像素，保持 H、W 不变。
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        # Sigmoid：生成范围在 (0, 1) 的空间位置权重。
        self.sigmoid = nn.Sigmoid()

    # 前向传播：输入 x 的维度为 [B, C, H, W]。
    def forward(self, x):
        # 在通道维 dim=1 求均值，获得每个空间位置的平均响应：[B, 1, H, W]。
        # keepdim=True 表示保留被归约的维度，方便后续拼接。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 在通道维 dim=1 求最大值，获得每个位置的最强响应：[B, 1, H, W]。
        # torch.max 同时返回最大值和索引，使用 _ 忽略索引。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均响应图和最大响应图在通道维拼接：[B, 2, H, W]。
        out = torch.cat([avg_out, max_out], dim=1)
        # 通过 3×3 卷积学习邻域信息，生成单通道空间权重图：[B, 1, H, W]。
        out = self.conv1(out)
        # 使用 Sigmoid 将各个空间位置的权重映射到 (0, 1)。
        out = self.sigmoid(out)
        # 返回空间注意力图（此处尚未与输入 x 相乘）。
        return out

# ========================= 第三部分：跨模态聚合（CrossModalAggregation） =========================
# 目的：自适应分配主/辅助模态的贡献，增强主模态后再融合辅助模态。
class CrossModalAggregation(nn.Module):
    # in_channels：每路输入特征的通道数；两路输入的通道数均应为 C。
    def __init__(self, in_channels):
        # 初始化父类 nn.Module。
        super(CrossModalAggregation, self).__init__()
        # 1×1 卷积：把两模态拼接后的 2C 通道映射成 2 通道权重图。
        # 输入 [B, 2C, H, W] → 输出 [B, 2, H, W]。
        self.conv = nn.Conv2d(in_channels * 2, 2, kernel_size=1, padding=0, bias=False)
        # Sigmoid：分别约束两路模态权重图的值到 (0, 1)。
        self.sigmoid = nn.Sigmoid()
        # 全局平均池化：将每张模态权重图汇总为一个标量权重。
        # 输入 [B, 1, H, W] → 输出 [B, 1, 1, 1]。
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 创建通道注意力模块，用于增强加权后的主模态特征。
        self.ChannelAttention = ChannelAttention(in_channels)
        # 创建空间注意力模块，用于进一步突出主模态的关键空间位置。
        self.SpatialAttention = SpatialAttention()

    # 前向传播：x_main 是主模态；x_supple 是辅助模态。
    # 两路输入均为 [B, C, H, W]，且 B、C、H、W 应完全一致。
    def forward(self, x_main, x_supple):
        # ---------- 步骤 1：拼接双模态特征，学习模态权重 ----------
        # 沿通道维拼接主模态和辅助模态：[B, C, H, W] × 2 → [B, 2C, H, W]。
        x = torch.cat([x_main, x_supple], dim=1)
        # 使用 1×1 卷积联合建模两路模态信息，并输出两张权重图：[B, 2, H, W]。
        # 注意：这是卷积生成的融合权重，不是显式的差分计算或 Q/K/V Cross Attention。
        x = self.conv(x)

        # ---------- 步骤 2：获得两路独立的模态权重 ----------
        # 取第 0 个通道作为主模态权重图，第 1 个通道作为辅助模态权重图。
        w_main, w_supple = x[:, 0:1, :, :], x[:, 1:2, :, :]
        # 将主模态权重图映射到 (0, 1)。
        w_main = self.sigmoid(w_main)
        # 将辅助模态权重图映射到 (0, 1)。
        w_supple = self.sigmoid(w_supple)

        # ---------- 步骤 3：将空间权重图压缩为每路模态的全局权重 ----------
        # 对主模态权重图做全局平均池化：[B, 1, H, W] → [B, 1, 1, 1]。
        w_main = self.pool(w_main)
        # 对辅助模态权重图做全局平均池化：[B, 1, H, W] → [B, 1, 1, 1]。
        w_supple = self.pool(w_supple)
        # 注意：这两步是空间池化，不是将两路权重归一化到和为 1。

        # ---------- 步骤 4：根据模态权重分别调节两路特征 ----------
        # 主模态乘以自身的全局标量权重（广播机制），维度保持 [B, C, H, W]。
        g_inp1 = w_main * x_main
        # 辅助模态乘以自身的全局标量权重，维度保持 [B, C, H, W]。
        g_inp2 = w_supple * x_supple

        # ---------- 步骤 5：主模态依次经过通道注意力和空间注意力 ----------
        # CA 首先输出 [B, C, 1, 1] 的通道权重，再与主模态特征逐元素相乘。
        # 依靠广播机制，使各通道整体增强或减弱；结果为 [B, C, H, W]。
        x_main_CA = self.ChannelAttention(g_inp1) * g_inp1
        # SA 输出 [B, 1, H, W] 的空间权重，再与经过 CA 的主模态逐元素相乘。
        # 依靠广播机制，使关键空间位置获得更大的权重；结果仍为 [B, C, H, W]。
        x_main_CA_SA = self.SpatialAttention(x_main_CA) * x_main_CA

        # ---------- 步骤 6：融合增强后的主模态与辅助模态 ----------
        # 将经过 CA+SA 增强的主模态与加权辅助模态逐元素相加。
        # 该融合方式对两路特征不对称：CA、SA 只作用于主模态。
        fuse = x_main_CA_SA + g_inp2
        # 返回跨模态融合特征，形状与输入保持一致：[B, C, H, W]。
        return fuse

# ========================= 第四部分：简单运行测试 =========================
if __name__ == "__main__":
    # 如果检测到 CUDA GPU，使用 cuda:0；否则退回 CPU。
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 构造第一路随机特征，作为主模态：[批大小=1, 通道=64, 高=32, 宽=32]。
    x1 = torch.randn(1, 64, 32, 32).to(device)
    x2 = torch.randn(1, 64, 32, 32).to(device)
    model = CrossModalAggregation(64).to(device)
    y = model(x1, x2)
    print("输入特征1维度：", x1.shape)
    print("输入特征2维度：", x2.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")
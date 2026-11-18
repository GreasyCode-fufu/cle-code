import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

"""
    论文地址：hhttps://arxiv.org/pdf/2503.23359
    论文题目：VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion(CVPR 2026)
    中文题目：VideoFusion：面向多模态视频融合的时空协同网络 (CVPR 2026)
    讲解视频：https://www.bilibili.com/video/BV1zhJC6zEUj/
    完整模态引导融合模块（Complete Modality-guided Fusion Module，CMGF）
        实际意义：①简单融合方式无法充分利用模态特异性信息：传统方法通常采用特征相加、特征拼接直接融合不同模态特征，能够整合红外和可见光的整体信息，但容易导致不同模态之间的信息混合，无法针对不同模态选择有效信息。②缺少有效的跨模态信息选择机制：冗余信息通常为两种模态共同存在的背景信息，需要根据融合需求，自适应提取不同模态的有效信息。
        核心思想：通过跨模态注意力自适应挖掘不同模态中的互补信息，实现更加完整有效的多模态特征融合。
        代码流程：输入红外与可见光特征 → 特征相加生成综合模态Query → 分别映射生成Key/Value → 跨模态Attention自适应提取有效信息 → FFN增强融合特征 → 输出完整融合表示
"""
# 4D特征转3D序列
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


# 3D序列转回4D特征
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 无偏置层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# 带偏置层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# 图像专用归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 跨模态交叉注意力
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm = LayerNorm(dim, LayerNorm_type)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_A, x_B):
        b, c, h, w = x_A.shape

        q = self.q_dwconv(self.q(x_A))
        kv = self.kv_dwconv(self.kv(x_B))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# 门控前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# 跨模态Transformer块
class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(Cross_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, supple):
        x = x + self.attn(self.norm1(x), self.norm1(supple))
        x = x + self.ffn(self.norm2(x))
        return x

# 双模态融合核心模块
class FusionModule(nn.Module):
    def __init__(self,
                 in_channels=192,
                 num_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 embed_dim=64,
                 group=4):
        super(FusionModule, self).__init__()
        self.cross_transformer = Cross_TransformerBlock(
            dim=in_channels,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            embed_dim=embed_dim,
            group=group)

    def forward(self, ir, vi):
        assert ir.shape == vi.shape
        Q = ir + vi
        fusion_ir = self.cross_transformer(Q, ir)
        fusion_vi = self.cross_transformer(Q, vi)
        fusion = fusion_ir + fusion_vi
        return fusion

if __name__ == "__main__":
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    model = FusionModule(64)
    y = model(x1, x2)
    print("输入特征1维度：", x1.shape)
    print("输入特征2维度：", x2.shape)
    print("输出特征维度：", y.shape)
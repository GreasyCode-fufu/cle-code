import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2603.27301
    论文题目：Dual-Path Learning based on Frequency Structural Decoupling and Regional-Aware Fusion for Low-Light Image Super-Resolution (ICME 2026)
    中文题目：基于频率结构解耦与区域感知融合的微光图像超分辨率双路径学习 (ICME 2026)
    讲解视频：https://www.bilibili.com/video/BV1Z5hx62EmV/
    跨频率语义重组模块（Cross-frequency Semantic Recomposition，CSR）
        实际意义：①静态融合无法适应不同区域的特征差异：传统方法常采用固定权重、直接相加或简单Concat 来融合不同特征，但图像不同区域对亮度信息和纹理信息的依赖程度并不相同。固定融合方式无法根据图像内容动态调整，容易造成局部结构退化和语义错位。
                ②不同区域需要动态决定融合重点：图像中有些区域更需要亮度校正，有些区域则更依赖纹理细节特征。
        实现方式：通过空间—通道双注意力与动态门控自适应增强低频亮度和高频纹理特征，并结合高频残差连接保留细节，实现结构一致的跨频率语义融合。
        代码流程：原始特征/纹理特征/亮度特征输入 → 卷积映射统一特征空间 → 通道+空间注意力进行频率感知增强 → Softmax自适应融合多源频率信息 
                    → 多尺度卷积提取上下文特征 → 门控混合与残差块细化语义 → 特征重构与PixelShuffle上采样 → 输出增强高分辨率特征。
"""

class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)
        return x + residual


class ChannelAttention(nn.Module):
    """通道注意力，使用平均池化和最大池化两路统计。"""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.sigmoid(pooled)


class SpatialAttention(nn.Module):
    """空间注意力，使用通道均值图和通道最大图生成注意力。"""
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attention)

class FrequencyAwareModule(nn.Module):
    """CSR 中的频率感知模块：分别对纹理和亮度特征施加通道/空间注意力。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(
        self,
        texture_features: torch.Tensor,
        luminance_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texture_features = self.channel_attention(texture_features) * texture_features
        luminance_features = self.channel_attention(luminance_features) * luminance_features
        texture_features = self.spatial_attention(texture_features) * texture_features
        luminance_features = self.spatial_attention(luminance_features) * luminance_features
        return texture_features, luminance_features


class AdaptiveFusion(nn.Module):
    """三输入自适应融合：原图特征、纹理特征、亮度特征按 softmax 权重加权。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        original: torch.Tensor,
        texture_features: torch.Tensor,
        luminance_features: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.attention(torch.cat([original, texture_features, luminance_features], dim=1))
        return (
            weights[:, [0], :, :] * original
            + weights[:, [1], :, :] * texture_features
            + weights[:, [2], :, :] * luminance_features
        )


class AdaptiveMixingModule(nn.Module):
    """特征混合模块：根据增强特征生成通道权重，在 base 和 enhanced 间插值。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(channels // 4, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, base: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        mixing_weight = self.attention(enhanced)
        return mixing_weight * enhanced + (1.0 - mixing_weight) * base

class AdaptiveLayerNorm(nn.Module):
    """对 NCHW 特征做 LayerNorm：临时转为 NHWC 后再转回。"""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x.permute(0, 2, 3, 1))
        return normalized.permute(0, 3, 1, 2).reshape(batch, channels, height, width)

class CrossFrequencySemanticRecomposition(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: int = 24,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        # 默认参数 in_channels=3、features=24、dropout_rate=0.1 来自原始 csr.py。
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_first = nn.Conv2d(in_channels, features, 3, 1, 1)
        self.frequency_aware = FrequencyAwareModule(features)
        self.adaptive_fusion = AdaptiveFusion(features)
        self.fusion = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            AdaptiveLayerNorm(features),
            nn.LeakyReLU(0.2, inplace=True),
            self.dropout,
            nn.Conv2d(features, features, 3, 1, 1),
            AdaptiveLayerNorm(features),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feature_mixing = AdaptiveMixingModule(features)
        # 多尺度 3x3/5x5/7x7 卷积，对应图中的 Multi-Scale Extraction and Fusion。
        self.multi_scale_fusion = nn.ModuleList(
            [
                nn.Conv2d(features, features, 3, 1, 1),
                nn.Conv2d(features, features, 5, 1, 2),
                nn.Conv2d(features, features, 7, 1, 3),
            ]
        )
        self.feature_gate = nn.Sequential(
            nn.Conv2d(features * 3, features, 1),
            nn.Sigmoid(),
        )
        self.body = nn.ModuleList([ResBlock(features) for _ in range(3)])
        self.reconstruction = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, 3, 1, 1),
            AdaptiveLayerNorm(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            self.dropout,
            ResBlock(features * 2),
            nn.Conv2d(features * 2, features * 2, 3, 1, 1),
            AdaptiveLayerNorm(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )


        self.upconv = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
            ResBlock(features),
            nn.Conv2d(features, features * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
            ResBlock(features),
            nn.Conv2d(features, in_channels, 3, 1, 1),
        )

    def forward(
        self,
        original_llr: torch.Tensor,
        texture_dlr: torch.Tensor,
        luminance_hlr: torch.Tensor,
    ) -> torch.Tensor:
        feat_input = self.conv_first(original_llr)
        feat_texture = self.conv_first(texture_dlr)
        feat_luminance = self.conv_first(luminance_hlr)

        # 频率感知注意力分别校准纹理和亮度特征。
        texture_aware, luminance_aware = self.frequency_aware(feat_texture, feat_luminance)
        fused = self.adaptive_fusion(feat_input, texture_aware, luminance_aware)
        features = self.fusion(fused)

        # 多尺度卷积后生成门控权重，再与输入特征做自适应混合。
        multi_scale = torch.cat([conv(features) for conv in self.multi_scale_fusion], dim=1)
        features = self.feature_mixing(feat_input, features * self.feature_gate(multi_scale))

        refined_features = features
        for block in self.body:
            refined_features = block(refined_features)

        # 拼接原图特征和细化特征，经过重建模块与 PixelShuffle 上采样。
        reconstruction = self.reconstruction(torch.cat([feat_input, refined_features], dim=1))
        restored_hsr = self.upconv(reconstruction)
        return torch.tanh(restored_hsr) * 0.5 + 0.5

if __name__ == "__main__":
    # 最小可运行示例：x2 超分输入 [B, 3, H, W]，输出 [B, 3, 2H, 2W]。
    original_llr = torch.rand(1, 3, 32, 32) # 原始特征
    texture_dlr = torch.rand(1, 3, 32, 32) # 纹理特征
    luminance_hlr = torch.rand(1, 3, 32, 32) # 亮度特征
    model = CrossFrequencySemanticRecomposition()
    restored_hsr = model(original_llr, texture_dlr, luminance_hlr)
    print("restored_hsr:", restored_hsr.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")

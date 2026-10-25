import torch
import torch.nn as nn


class DBFE(nn.Module):
    """Dual-Branch Feature Enhancement from TPSegformer.

    The module combines an ECANet-style channel descriptor with a 3x3 local
    spatial branch. The channel branch decides which semantic responses matter;
    the spatial branch restores local texture and boundary cues before decoding.
    """

    def __init__(self, channel, k_size=3, dropout=0.2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        channel_descriptor = self.avg_pool(x)
        channel_weight = self.channel_conv(
            channel_descriptor.squeeze(-1).transpose(-1, -2)
        )
        channel_weight = channel_weight.transpose(-1, -2).unsqueeze(-1)
        channel_weight = self.sigmoid(channel_weight)

        spatial_feature = self.spatial_branch(x)
        return spatial_feature * channel_weight.expand_as(spatial_feature)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = DBFE(64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape :", input_tensor.shape)
    print("output_tensor_shape:", output_tensor.shape)
    print("哔哩哔哩/微信公众号 CV缝合救星 独家整理!")

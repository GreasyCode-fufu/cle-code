import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RegionAwareAAPB(nn.Module):
    """Region-aware AAPB with a learnable reliability gate.

    Magic change: the original AAPB uses one sample-level adaptive coefficient.
    This version keeps the same score tensor contract, then predicts a spatial
    confidence map so rare concepts, edited objects, and background regions can
    receive different anchor-target balances.
    """

    def __init__(self, channels: int, guidance_scale: float = 7.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.eps = eps
        self.reliability = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def global_gamma(self, uncond_score: torch.Tensor, target_score: torch.Tensor, anchor_score: torch.Tensor) -> torch.Tensor:
        direction = anchor_score - target_score
        residual = target_score - uncond_score
        numerator = -(residual * direction).flatten(1).sum(dim=1)
        denominator = direction.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        return (numerator / denominator).clamp(0.0, 1.0).view(-1, 1, 1, 1)

    def forward(
        self,
        uncond_score: torch.Tensor,
        target_score: torch.Tensor,
        anchor_score: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        base_gamma = self.global_gamma(uncond_score, target_score, anchor_score)
        disagreement = torch.cat(
            [
                (target_score - uncond_score).abs(),
                (anchor_score - uncond_score).abs(),
                (target_score - anchor_score).abs(),
            ],
            dim=1,
        )
        local_reliability = self.reliability(disagreement)
        local_gamma = torch.sigmoid((base_gamma - 0.5) * self.temperature) * local_reliability
        local_gamma = F.avg_pool2d(local_gamma, 3, stride=1, padding=1)
        conditional = (1.0 - local_gamma) * target_score + local_gamma * anchor_score
        return uncond_score + scale * (conditional - uncond_score)


def main() -> None:
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟扩散模型 score/latent 输入 [B, C, H, W]
    input_tensor = torch.randn(1, 4, 32, 32).to(device)
    target_tensor = input_tensor + 0.2 * torch.randn_like(input_tensor)
    anchor_tensor = input_tensor + 0.2 * torch.randn_like(input_tensor)

    # 直接实例化真正的核心创新模块：RA-AAPB
    model = RegionAwareAAPB(channels=4, guidance_scale=6.5).to(device)

    print(model)

    output_tensor = model(input_tensor, target_tensor, anchor_tensor)
    loss = output_tensor.square().mean()
    loss.backward()

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")


if __name__ == "__main__":
    main()

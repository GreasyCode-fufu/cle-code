import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class StableCPCAChannelAttention(nn.Module):
    """
    支持三种通道门控方式：

    legacy:
        sigmoid(MLP(avg)) + sigmoid(MLP(max))
        与用户当前代码保持一致，权重范围约为 (0, 2)。

    paper:
        sigmoid(MLP(avg) + MLP(max))
        与原始 CPCA 论文公式一致，权重范围为 (0, 1)。

    gain:
        2 * sigmoid(MLP(avg) + MLP(max))
        保留 (0, 2) 的增益范围，同时使零响应对应单位增益 1。
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        gate_mode: str = "legacy",
    ):
        super().__init__()

        if reduction <= 0:
            raise ValueError("reduction must be positive.")

        if gate_mode not in {"legacy", "paper", "gain"}:
            raise ValueError(
                "gate_mode must be 'legacy', 'paper', or 'gain'."
            )

        hidden_channels = max(1, channels // reduction)

        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.gate_mode = gate_mode

    def shared_mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = F.adaptive_avg_pool2d(x, output_size=1)
        max_descriptor = F.adaptive_max_pool2d(x, output_size=1)

        avg_logits = self.shared_mlp(avg_descriptor)
        max_logits = self.shared_mlp(max_descriptor)

        if self.gate_mode == "legacy":
            channel_gate = (
                torch.sigmoid(avg_logits)
                + torch.sigmoid(max_logits)
            )

        elif self.gate_mode == "paper":
            channel_gate = torch.sigmoid(
                avg_logits + max_logits
            )

        else:  # gain
            channel_gate = 2.0 * torch.sigmoid(
                avg_logits + max_logits
            )

        return x * channel_gate


class RoleAwarePointwiseProjection(nn.Module):
    """
    共享 1x1 卷积 + 不同角色的低秩适配器。

    与直接使用三个独立 C×C 卷积相比，增加的参数量较少。
    每个适配器的最后一层采用零初始化，因此初始化时：

        output = shared_conv(x)

    与原来重复调用同一个 self.conv 的行为保持一致。
    """

    def __init__(
        self,
        channels: int,
        adapter_reduction: int = 16,
    ):
        super().__init__()

        adapter_channels = max(
            1,
            channels // adapter_reduction,
        )

        self.shared_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.adapters = nn.ModuleDict()

        for role in ("input", "attention", "output"):
            adapter = nn.Sequential(
                nn.Conv2d(
                    channels,
                    adapter_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(
                    adapter_channels,
                    channels,
                    kernel_size=1,
                    bias=False,
                ),
            )

            # 使角色适配器初始输出为 0
            nn.init.zeros_(adapter[-1].weight)
            self.adapters[role] = adapter

    def forward(
        self,
        x: torch.Tensor,
        role: str,
    ) -> torch.Tensor:
        if role not in self.adapters:
            raise ValueError(f"Unknown projection role: {role}")

        shared_feature = self.shared_conv(x)
        role_correction = self.adapters[role](x)

        return shared_feature + role_correction


class PerChannelScaleRouter(nn.Module):
    """
    为每个样本、每个通道预测不同的多尺度分支权重。

    输出权重尺寸：
        B × K × C × 1 × 1

    最后一层采用零初始化，初始 logits 全为 0：
        softmax(logits) = 1 / K

    再乘以 K 后，每个分支的初始权重为 1，
    因此初始化时等价于原始的固定分支求和。
    """

    def __init__(
        self,
        channels: int,
        num_branches: int = 4,
        reduction: int = 8,
    ):
        super().__init__()

        hidden_channels = max(
            4,
            channels // reduction,
        )

        self.channels = channels
        self.num_branches = num_branches

        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            hidden_channels,
            num_branches * channels,
            kernel_size=1,
            bias=True,
        )

        # 保证初始化时每个分支权重都等于 1
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        context: torch.Tensor,
        branches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(branches) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branches, "
                f"but got {len(branches)}."
            )

        batch_size, channels, _, _ = context.shape

        descriptor = F.adaptive_avg_pool2d(
            context,
            output_size=1,
        )
        descriptor = F.gelu(self.fc1(descriptor))

        logits = self.fc2(descriptor)
        logits = logits.view(
            batch_size,
            self.num_branches,
            channels,
            1,
            1,
        )

        # 沿分支维度归一化
        scale_weights = torch.softmax(
            logits,
            dim=1,
        )

        # 初始时每个分支权重从 1/K 恢复到 1
        scale_weights = (
            scale_weights * self.num_branches
        )

        stacked_branches = torch.stack(
            branches,
            dim=1,
        )

        fused_feature = (
            stacked_branches * scale_weights
        ).sum(dim=1)

        return fused_feature, scale_weights


class CPCA2(nn.Module):
    """
    Adaptive Scale-Routed CPCA

    改进点：
    1. 保留原 CPCA 的通道先验和多尺度条带卷积；
    2. 使用逐通道动态尺度路由替代固定求和；
    3. 使用低秩角色适配器缓解三个 1x1 卷积位置的任务冲突；
    4. 使用零初始化的高频边界补偿分支；
    5. 所有新增路径在初始化时均退化为原 CPCA。
    """

    def __init__(
        self,
        channels: int,
        channel_reduction: int = 4,
        router_reduction: int = 8,
        adapter_reduction: int = 16,
        channel_gate_mode: str = "legacy",
        use_boundary_branch: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.use_boundary_branch = use_boundary_branch

        self.channel_attention = StableCPCAChannelAttention(
            channels=channels,
            reduction=channel_reduction,
            gate_mode=channel_gate_mode,
        )

        # 原始局部深度卷积
        self.dconv5_5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True,
        )

        # 7×7 条带卷积分支
        self.dconv1_7 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
            bias=True,
        )
        self.dconv7_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
            bias=True,
        )

        # 11×11 条带卷积分支
        self.dconv1_11 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 11),
            padding=(0, 5),
            groups=channels,
            bias=True,
        )
        self.dconv11_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(11, 1),
            padding=(5, 0),
            groups=channels,
            bias=True,
        )

        # 21×21 条带卷积分支
        self.dconv1_21 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 21),
            padding=(0, 10),
            groups=channels,
            bias=True,
        )
        self.dconv21_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(21, 1),
            padding=(10, 0),
            groups=channels,
            bias=True,
        )

        self.scale_router = PerChannelScaleRouter(
            channels=channels,
            num_branches=4,
            reduction=router_reduction,
        )

        self.pointwise_projection = (
            RoleAwarePointwiseProjection(
                channels=channels,
                adapter_reduction=adapter_reduction,
            )
        )

        self.activation = nn.GELU()

        if use_boundary_branch:
            self.boundary_conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            )

            # 初始化时边界分支完全关闭
            nn.init.zeros_(self.boundary_conv.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        return_scale_weights: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        # 1. 输入特征投影
        x = self.pointwise_projection(
            inputs,
            role="input",
        )
        x = self.activation(x)

        # 2. 构建通道先验
        channel_prior = self.channel_attention(x)

        # 3. 原 CPCA 多尺度空间分支
        x_init = self.dconv5_5(channel_prior)

        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)

        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)

        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)

        # 4. 逐样本、逐通道动态尺度融合
        spatial_feature, scale_weights = (
            self.scale_router(
                context=channel_prior,
                branches=[
                    x_init,
                    x_1,
                    x_2,
                    x_3,
                ],
            )
        )

        # 5. 高频边界补偿
        if self.use_boundary_branch:
            local_smooth = F.avg_pool2d(
                channel_prior,
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            )

            high_frequency = (
                channel_prior - local_smooth
            )

            boundary_feature = self.boundary_conv(
                high_frequency
            )

            spatial_feature = (
                spatial_feature + boundary_feature
            )

        # 6. 注意力生成位置的角色适配
        spatial_attention = (
            self.pointwise_projection(
                spatial_feature,
                role="attention",
            )
        )

        # 保留原 CPCA 的非 Sigmoid 空间调制形式
        refined_feature = (
            spatial_attention * channel_prior
        )

        # 7. 输出重构位置的角色适配
        output = self.pointwise_projection(
            refined_feature,
            role="output",
        )

        if return_scale_weights:
            return output, scale_weights

        return output

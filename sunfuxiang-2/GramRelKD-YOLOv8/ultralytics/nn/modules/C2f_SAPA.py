from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Ultralytics components
# ============================================================

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1
            if isinstance(k, int)
            else [d * (x - 1) + 1 for x in k]
        )

    if p is None:
        p = (
            k // 2
            if isinstance(k, int)
            else [x // 2 for x in k]
        )

    return p


class Conv(nn.Module):
    """Standard convolution + BN + activation."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard YOLOv8 Bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=(3, 3),
        e: float = 0.5
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(
            c1,
            c_,
            k[0],
            1
        )

        self.cv2 = Conv(
            c_,
            c2,
            k[1],
            1,
            g=g
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


# ============================================================
# Utility
# ============================================================

def _best_num_heads(dim: int, requested_heads: int) -> int:
    """
    Automatically find a valid number of heads.

    Example:
        dim=32, requested_heads=4 -> 4
        dim=30, requested_heads=4 -> 3
    """

    heads = min(requested_heads, dim)

    while heads > 1 and dim % heads != 0:
        heads -= 1

    return heads


# ============================================================
# Scale-Aware Partial Attention for 2D feature maps
# ============================================================

class WindowScaleAwarePartialAttention2D(nn.Module):
    """
    Window-based Scale-Aware Partial Attention.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]

    Core idea:
        1. Split channels into active / frozen parts.
        2. Only active channels perform attention.
        3. Use global mean + variance to estimate scale context.
        4. Add lightweight local depthwise convolution.
        5. Frozen channels retain a lightweight gated shortcut.
        6. Use window attention instead of global attention to control cost.
    """

    def __init__(
        self,
        dim: int,
        ratio: float = 0.5,
        num_heads: int = 4,
        window_size: int = 8,
        reduction: int = 4,
        local_kernel: int = 3,
        qkv_bias: bool = False
    ):
        super().__init__()

        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")

        self.dim = dim

        # ----------------------------------------------------
        # 1. Partial channel selection
        # ----------------------------------------------------
        self.active_dim = max(
            1,
            min(dim, int(round(dim * ratio)))
        )

        self.frozen_dim = dim - self.active_dim

        # Automatically guarantee divisibility
        self.num_heads = _best_num_heads(
            self.active_dim,
            num_heads
        )

        self.head_dim = (
            self.active_dim // self.num_heads
        )

        self.scale = self.head_dim ** -0.5

        self.window_size = window_size

        # ----------------------------------------------------
        # 2. Attention branch
        # ----------------------------------------------------
        self.norm = nn.LayerNorm(
            self.active_dim
        )

        self.qkv = nn.Linear(
            self.active_dim,
            self.active_dim * 3,
            bias=qkv_bias
        )

        self.proj = nn.Linear(
            self.active_dim,
            self.active_dim
        )

        # ----------------------------------------------------
        # 3. Scale-aware gate
        #
        # Mean:
        # global activation level
        #
        # Variance:
        # spatial fluctuation / fine-detail information
        # ----------------------------------------------------
        hidden = max(
            8,
            self.active_dim // reduction
        )

        self.scale_gate = nn.Sequential(

            nn.Linear(
                self.active_dim * 2,
                hidden
            ),

            nn.GELU(),

            nn.Linear(
                hidden,
                self.active_dim
            ),

            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # 4. Local detail branch
        #
        # Depthwise convolution:
        # compensate for local information
        # ----------------------------------------------------
        padding = local_kernel // 2

        self.local_mixer = nn.Conv2d(
            self.active_dim,
            self.active_dim,
            kernel_size=local_kernel,
            padding=padding,
            groups=self.active_dim,
            bias=False
        )

        # ----------------------------------------------------
        # 5. Frozen-channel lightweight gate
        # ----------------------------------------------------
        if self.frozen_dim > 0:

            self.frozen_gate = nn.Sequential(

                nn.LayerNorm(
                    self.frozen_dim
                ),

                nn.Linear(
                    self.frozen_dim,
                    self.frozen_dim
                ),

                nn.Sigmoid()
            )

        else:
            self.frozen_gate = None

        # Learnable local residual strength
        self.res_scale = nn.Parameter(
            torch.tensor(0.1)
        )


    # ========================================================
    # Window Attention
    # ========================================================

    def _window_attention(self, x):

        """
        x:
            [B, C_active, H, W]
        """

        b, c, h, w = x.shape

        # Avoid window larger than feature map
        ws = min(
            self.window_size,
            h,
            w
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------
        pad_h = (
            ws - h % ws
        ) % ws

        pad_w = (
            ws - w % ws
        ) % ws

        x_pad = F.pad(
            x,
            (0, pad_w, 0, pad_h)
        )

        hp, wp = x_pad.shape[-2:]

        num_h = hp // ws
        num_w = wp // ws

        # ----------------------------------------------------
        # [B,C,H,W]
        # ->
        # [B*nWindow, window², C]
        # ----------------------------------------------------
        tokens = (
            x_pad
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                b,
                num_h,
                ws,
                num_w,
                ws,
                c
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
                5
            )
            .contiguous()
            .view(
                b * num_h * num_w,
                ws * ws,
                c
            )
        )

        # ----------------------------------------------------
        # Q K V
        # ----------------------------------------------------
        tokens_norm = self.norm(tokens)

        qkv = self.qkv(tokens_norm)

        qkv = qkv.view(
            tokens.shape[0],
            tokens.shape[1],
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q, k, v = qkv.unbind(0)

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attn = attn.softmax(
            dim=-1
        )

        out = (
            attn @ v
        )

        out = (
            out
            .transpose(1, 2)
            .reshape(
                tokens.shape[0],
                tokens.shape[1],
                c
            )
        )

        out = self.proj(out)

        # ----------------------------------------------------
        # Window reverse
        # ----------------------------------------------------
        out = (
            out
            .view(
                b,
                num_h,
                num_w,
                ws,
                ws,
                c
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
                5
            )
            .contiguous()
            .view(
                b,
                hp,
                wp,
                c
            )
            .permute(
                0,
                3,
                1,
                2
            )
            .contiguous()
        )

        # Remove padding
        out = out[:, :, :h, :w]

        return out


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        """
        x:
            [B,C,H,W]
        """

        # ----------------------------------------------------
        # Channel split
        # ----------------------------------------------------
        active = x[
            :,
            :self.active_dim,
            :,
            :
        ]

        frozen = x[
            :,
            self.active_dim:,
            :,
            :
        ]

        # ----------------------------------------------------
        # Active branch:
        # window attention
        # ----------------------------------------------------
        attn_out = self._window_attention(
            active
        )

        # ----------------------------------------------------
        # Scale-aware descriptor
        #
        # LN across channel dimension
        # ----------------------------------------------------
        norm_active = self.norm(
            active.permute(
                0,
                2,
                3,
                1
            )
        )

        # Global mean
        token_mean = norm_active.mean(
            dim=(1, 2)
        )

        # Global variance
        token_var = norm_active.var(
            dim=(1, 2),
            unbiased=False
        )

        # ----------------------------------------------------
        # Scale gate
        # ----------------------------------------------------
        scale_descriptor = torch.cat(
            [
                token_mean,
                token_var
            ],
            dim=-1
        )

        gate = self.scale_gate(
            scale_descriptor
        )

        gate = gate.view(
            x.shape[0],
            self.active_dim,
            1,
            1
        )

        # ----------------------------------------------------
        # Local branch
        # ----------------------------------------------------
        local_out = self.local_mixer(
            active
        )

        # ----------------------------------------------------
        # Active feature fusion
        # ----------------------------------------------------
        active_out = (

            active

            + attn_out * gate

            + self.res_scale * local_out
        )

        # ----------------------------------------------------
        # Frozen branch
        # ----------------------------------------------------
        if self.frozen_dim > 0:

            frozen_tokens = frozen.permute(
                0,
                2,
                3,
                1
            )

            frozen_weight = self.frozen_gate(
                frozen_tokens
            )

            frozen_out = (
                frozen_tokens
                * frozen_weight
            )

            frozen_out = frozen_out.permute(
                0,
                3,
                1,
                2
            ).contiguous()

            return torch.cat(
                [
                    active_out,
                    frozen_out
                ],
                dim=1
            )

        return active_out


# ============================================================
# Scale-Aware Partial C2f
# ============================================================

class C2f_SAPA(nn.Module):
    """
    Scale-Aware Partial Attention C2f.

    Original C2f:

                    Input
                      |
                     cv1
                      |
                    Split
                   /     \\
                  /       \\
            shortcut    Bottleneck
                           |
                      Bottleneck...
                   \\       /
                    Concat
                      |
                     cv2


    C2f_SAPA:

                    Input
                      |
                     cv1
                      |
                    Split
                  /       \\
                 /         \\
       Scale-aware       Semantic
       partial branch     branch
            |               |
         SAPA         Bottleneck × n
                 \\         /
                    Concat
                      |
                     cv2
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,

        # SAPA parameters
        ratio: float = 0.5,
        num_heads: int = 4,
        window_size: int = 8,
        reduction: int = 4
    ):
        super().__init__()

        self.c = int(c2 * e)

        # ----------------------------------------------------
        # Initial channel projection
        # ----------------------------------------------------
        self.cv1 = Conv(
            c1,
            2 * self.c,
            1,
            1
        )

        # ----------------------------------------------------
        # Scale-aware partial branch
        # ----------------------------------------------------
        self.sapa = (
            WindowScaleAwarePartialAttention2D(
                dim=self.c,
                ratio=ratio,
                num_heads=num_heads,
                window_size=window_size,
                reduction=reduction
            )
        )

        # ----------------------------------------------------
        # Standard C2f Bottleneck branch
        # ----------------------------------------------------
        self.m = nn.ModuleList(

            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0
            )

            for _ in range(n)
        )

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------
        self.cv2 = Conv(
            (2 + n) * self.c,
            c2,
            1
        )


    def forward(self, x):

        # Initial projection
        y0, y1 = self.cv1(x).chunk(
            2,
            dim=1
        )

        # ----------------------------------------------
        # y0:
        # scale-aware structural branch
        #
        # y1:
        # standard semantic branch
        # ----------------------------------------------
        y = [
            self.sapa(y0),
            y1
        ]

        # Standard C2f progressive Bottleneck
        y.extend(
            m(y[-1])
            for m in self.m
        )

        # Feature aggregation
        return self.cv2(
            torch.cat(
                y,
                dim=1
            )
        )


    def forward_split(self, x):

        y0, y1 = self.cv1(x).split(
            (
                self.c,
                self.c
            ),
            dim=1
        )

        y = [
            self.sapa(y0),
            y1
        ]

        y.extend(
            m(y[-1])
            for m in self.m
        )

        return self.cv2(
            torch.cat(
                y,
                dim=1
            )
        )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    x = torch.randn(
        2,
        128,
        80,
        80
    )

    model = C2f_SAPA(
        c1=128,
        c2=128,
        n=3,
        shortcut=True,

        ratio=0.5,
        num_heads=4,
        window_size=8
    )

    y = model(x)

    print(model)

    print(
        "Input shape :",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        params
    )
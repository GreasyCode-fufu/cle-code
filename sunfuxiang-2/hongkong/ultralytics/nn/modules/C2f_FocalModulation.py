import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


 
 

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
 
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
 
        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
 
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
 
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)
 
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]
 
        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)
 
        # context aggreation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
 
        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
 
        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
 
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out
    

class C2f_FocalModulation(nn.Module):
    """C2f with FocalModulation module integrated"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.fm = FocalModulation(dim=2*self.c)  # 添加焦点调制模块
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 通过焦点调制处理特征
        x = self.fm(self.cv1(x))  # 先卷积后调制
        
        # 分割特征图
        y = list(x.chunk(2, 1))  # 沿通道维度分割为两部分
        
        # 通过Bottleneck处理第二部分特征
        y.extend(m(y[-1]) for m in self.m)
        
        # 拼接并输出最终结果
        return self.cv2(torch.cat(y, 1))
"""
CLE-adapted ConvNeXt  (cle_convnext)
====================================
Baseline : models/Convnext.py  (convnext_tiny, 27.8 M, 4.45 GMACs @224)
Target   : confocal laser endomicroscopy (CLE) 3-class classification
           (Benign_lesions / Malignant_lesions / body_of_stomach_normal)

Three complementary inductive biases, matched to CLE physics
-----------------------------------------------------------
I1  FADES  Frequency-Adaptive Detail Excitation Stem
    CLE frames are low-contrast, single-channel (pseudo-colour) fluorescence
    images whose diagnosis lives in *high-frequency micro-texture* (pit
    patterns, tortuous vessels, crypt openings), whereas the ConvNeXt 4x4
    stride-4 patchify stem is a low-pass operator that throws that away.
    FADES computes fixed Sobel-x / Sobel-y / Laplacian responses per channel,
    turns them into a per-channel excitation map with a 2-layer bottleneck,
    and adds it back multiplicatively *before* patchify.  No extra
    down-sampling, shapes stay aligned, cost is ~4 k parameters.

I2  ADB    Anisotropic Dilated Block
    Ducts / vessels in CLE are *elongated, directional* structures, while the
    stock block uses a single isotropic 7x7 depth-wise conv.  ADB keeps that
    7x7 conv (so the ImageNet weights still load, key name unchanged) and adds
    two orthogonal strip convs (1x7, 7x1) plus a dilated 3x3 (d=2), fused by a
    content-adaptive channel gate with *zero-initialised* per-channel scales.
    At initialisation ADB == stock block exactly, so fine-tuning starts from
    the pretrained baseline instead of a randomly perturbed one.

I3  DPGH   Dual-Path Gated Head
    The stock head only sees stage-4 semantics (stride 32).  CLE lesions are
    small and the discriminative margin/texture detail lives in stage-2
    (stride 8).  DPGH pools both, projects to a shared 256-d space and lets a
    channel gate decide how much detail vs. semantics to trust per sample.

Every switch is independently ablatable:
    cle_convnext_v1        all three on
    cle_convnext_nofreq    I2 + I3
    cle_convnext_noaniso   I1 + I3
    cle_convnext_nodual    I1 + I2

Design constraint: `train_cle_fair.load_pretrained_weights_strict` audits the
state dict.  We therefore keep every pretrained tensor at its ORIGINAL key
(stages.i.j.dwconv.*, norm, pwconv1/2, gamma) and only *add* new parameters,
whose root segments are declared in MODEL_ALLOWED_MISSING_SEGMENTS.
"""
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import Block, DropPath, LayerNorm


# --------------------------------------------------------------------------- #
# I1: Frequency-Adaptive Detail Excitation Stem
# --------------------------------------------------------------------------- #
class FreqExcite(nn.Module):
    """Fixed high-pass banks -> learned per-channel excitation, added to input.

    Args:
        in_ch: input channels (3 for RGB / pseudo-colour CLE).
        hid:   bottleneck width of the gate MLP (kept tiny on purpose).
    """

    def __init__(self, in_ch: int = 3, hid: int = 16, mode: str = "sobel"):
        super().__init__()
        assert mode in ("sobel", "dwt", "both"), mode
        self.mode = mode
        # ---- fixed high-pass bank: 1st-order Sobel x/y + 2nd-order Laplacian
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t().contiguous()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        bank = torch.stack([sobel_x, sobel_y, lap], dim=0).unsqueeze(1)   # (3,1,3,3)
        # depth-wise layout: output channel k*C+c applies filter k to input channel c
        bank = bank.repeat_interleave(in_ch, dim=0)                       # (3*C,1,3,3)
        self.register_buffer("bank", bank, persistent=False)

        # ---- Haar analysis bank (orthonormal, stride 2) for the optional DWT branch
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        haar = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)          # (4,1,2,2)
        self.register_buffer("haar", haar.repeat_interleave(in_ch, dim=0), persistent=False)

        # channel count of the high-frequency descriptor fed to the gate
        n_desc = {"sobel": 3 * in_ch, "dwt": in_ch, "both": 4 * in_ch}[mode]
        self.gate = nn.Sequential(
            nn.Conv2d(n_desc, hid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, in_ch, 1, bias=True),
        )
        # zero-init last conv -> spatially constant gate at init -> stem+LN sees the
        # same (scale-invariant) signal as the pretrained baseline
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _haar_energy(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        coef = F.conv2d(x, self.haar, stride=2, groups=c)          # (B,4C,H/2,W/2)
        _, lh, hl, hh = coef.split(c, dim=1)
        e = lh.abs() + hl.abs() + hh.abs()                         # (B,C,H/2,W/2)
        return F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        parts = []
        if self.mode in ("sobel", "both"):
            parts.append(F.conv2d(x, self.bank, padding=1, groups=c))   # (B,3C,H,W)
        if self.mode in ("dwt", "both"):
            parts.append(self._haar_energy(x))                          # (B,C,H,W)
        r = torch.cat(parts, dim=1)
        g = torch.sigmoid(self.gate(r))                                # (B,C,H,W)
        return x * (1.0 + g)                                           # multiplicative detail boost


# NOTE on key-name preservation
# -----------------------------
# `nn.Sequential` keeps the stock stem at `downsample_layers.0.0` / `.1`, so the
# pretrained tensors still match.  The excitation module is attached as the
# extra attribute `.freq` of that Sequential, which yields brand-new keys
# (`downsample_layers.0.freq.*`) that the strict loader is told to expect.
# Wrapping the stem in a custom Module instead would rename the conv to
# `downsample_layers.0.conv.weight` and break every pretrained load.


# --------------------------------------------------------------------------- #
# I2: Anisotropic Dilated Block
# --------------------------------------------------------------------------- #
class AnisoDilatedBlock(nn.Module):
    """ConvNeXt Block whose depth-wise stage is enriched anisotropically.

    Keeps `.dwconv` / `.norm` / `.pwconv1` / `.pwconv2` / `.gamma` / `.drop_path`
    at the original attribute paths so ImageNet weights load unchanged, and the
    new branches enter through zero-initialised per-channel scales => at step 0
    the block is *bit-wise equivalent* to the pretrained ConvNeXt block.
    """

    def __init__(self, dim: int, drop_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 k: int = 7, dil: int = 2, gate_ratio: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
        self.strip_h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim)
        self.strip_v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim)
        self.dil = nn.Conv2d(dim, dim, kernel_size=3, padding=dil, dilation=dil, groups=dim)
        hid = max(8, dim // gate_ratio)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hid, 1), nn.GELU(), nn.Conv2d(hid, dim, 1), nn.Sigmoid(),
        )
        # per-channel scales for the three new branches, init 0 => identity
        self.alpha = nn.Parameter(torch.zeros(3, dim))

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        base = self.dwconv(x)
        a = torch.tanh(self.alpha).view(3, 1, -1, 1, 1)
        branch = (a[0] * self.strip_h(x) + a[1] * self.strip_v(x) + a[2] * self.dil(x))
        x = base + self.gate(x) * branch

        x = x.permute(0, 2, 3, 1)                     # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# --------------------------------------------------------------------------- #
# I3: Dual-Path Gated Head
# --------------------------------------------------------------------------- #
class DualPathHead(nn.Module):
    """Detail (stage2) + semantic (stage4) gated fusion classifier."""

    def __init__(self, c_detail: int, c_semantic: int, num_classes: int,
                 width: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.proj_d = nn.Sequential(nn.Linear(c_detail, width), nn.LayerNorm(width))
        self.proj_s = nn.Sequential(nn.Linear(c_semantic, width), nn.LayerNorm(width))
        self.gate = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width), nn.Sigmoid()
        )
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, z_detail: torch.Tensor, z_semantic: torch.Tensor) -> torch.Tensor:
        d = self.proj_d(z_detail)
        s = self.proj_s(z_semantic)
        g = self.gate(torch.cat([d, s], dim=1))
        z = g * d + (1.0 - g) * s
        return self.fc(self.drop(z))


class ResidualDetailHead(nn.Module):
    """I3-v2: keep the baseline 768->C head, inject stage-2/3 detail residually.

    Why v1 (DualPathHead) under-performed: projecting the 768-d semantic vector
    down to 256-d and *mixing* it with the detail path throws away head capacity
    on a 2k-image dataset (ablation: nodual 0.9850 vs v1 0.9763).  Here the
    semantic path keeps the baseline width and key path, and the detail path
    enters through a zero-initialised scalar gate, so at step 0 this head is
    bit-wise the baseline classifier.
    """

    def __init__(self, c_detail: int, c_semantic: int, num_classes: int, p_drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(c_semantic, eps=1e-6)
        self.proj_d = nn.Linear(c_detail, c_semantic)
        self.gate = nn.Sequential(
            nn.Linear(2 * c_semantic, max(16, c_semantic // 4)), nn.GELU(),
            nn.Linear(max(16, c_semantic // 4), c_semantic), nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(c_semantic, num_classes)

    def forward(self, z_detail: torch.Tensor, z_semantic: torch.Tensor) -> torch.Tensor:
        z = self.norm(z_semantic)
        d = self.proj_d(z_detail)
        g = self.gate(torch.cat([z, d], dim=1))
        return self.fc(self.drop(z + torch.tanh(self.beta) * g * d))


class FOVMask(nn.Module):
    """Circular field-of-view prior for probe-based CLE.

    Every CLE frame is a disc: ~21% of a 224x224 square is black vignette.  Plain
    global average pooling therefore averages real mucosa together with dead
    pixels, and the ratio of dead area varies with probe contact.  This module
    builds a fixed feathered disc mask and provides masked average pooling at any
    feature resolution, so every pooled descriptor is computed over the FOV only.
    """

    def __init__(self, size: int = 224, feather: float = 0.06):
        super().__init__()
        yy = torch.linspace(-1, 1, size).view(-1, 1).expand(size, size)
        xx = torch.linspace(-1, 1, size).view(1, -1).expand(size, size)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        # smooth 0..1 ramp from r=1-feather to r=1
        m = torch.clamp((1.0 - r) / feather, 0.0, 1.0)
        self.register_buffer("mask", m.view(1, 1, size, size), persistent=False)

    def at(self, h: int, w: int) -> torch.Tensor:
        return F.interpolate(self.mask, size=(h, w), mode="bilinear", align_corners=False)

    def masked_avg(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        m = self.at(h, w)
        num = (x * m).sum(dim=[-2, -1])
        den = m.sum().clamp(min=1e-6)
        return num / den


def _sqrt_newton_schulz(A: torch.Tensor, num_iter: int = 3) -> torch.Tensor:
    """Batched matrix square root (iSQRT-COV / MPN-COV style).

    Clearing the architecture is what the Miami criteria grade: normal mucosa has
    a *regular* villiform/crypt pattern, neoplasia has irregular crypts, dilated
    capillaries and finally loss of architecture.  "Irregularity vs regularity"
    is a second-order property of the feature distribution, which plain global
    average pooling (a first-order statistic) discards.
    """
    d = A.shape[-1]
    eye = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
    normA = A.norm(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
    Y = A / normA
    Z = eye.clone()
    for _ in range(num_iter):
        # canonical Newton-Schulz iteration for the matrix square root:
        #   Y_{k+1} = Y_k (3I - Z_k Y_k) / 2 ,  Z_{k+1} = (3I - Z_k Y_k) Z_k / 2
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * normA.sqrt()


class CovPoolHead(nn.Module):
    """I5: second-order (covariance) texture statistics for architecture grading.

    stage-4 features (B,768,H,W) -> 1x1 conv to `dim` -> covariance over spatial
    positions -> trace-normalised matrix square root -> LayerNorm -> classifier.
    Concatenated with the first-order (GAP) descriptor so the head sees both
    "what is present" and "how heterogeneous it is".
    """

    def __init__(self, c_in: int, c_sem: int, num_classes: int,
                 dim: int = 96, p_drop: float = 0.2, sqrt_iters: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.dim = dim
        self.sqrt_iters = sqrt_iters
        self.norm = nn.LayerNorm(dim * dim + c_sem, eps=1e-6)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(dim * dim + c_sem, num_classes)

    def forward(self, x_spatial: torch.Tensor, z_semantic: torch.Tensor) -> torch.Tensor:
        f = self.proj(x_spatial)                       # (B,d,H,W)
        b, d, h, w = f.shape
        n = h * w
        x = f.view(b, d, n)
        x = x - x.mean(dim=2, keepdim=True)
        cov = x @ x.transpose(1, 2) / max(n - 1, 1)
        cov = cov + 1e-5 * torch.eye(d, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1).view(b, 1, 1).clamp(min=1e-6)
        cov = cov / tr
        s = _sqrt_newton_schulz(cov, self.sqrt_iters).reshape(b, d * d)
        z = torch.cat([s, z_semantic], dim=1)
        return self.fc(self.drop(self.norm(z)))


class MultiExitHead(nn.Module):
    """I6: aggregated multi-exit classifier (implicit deep supervision).

    `train_cle_fair.py` applies exactly one CE loss to model(x), so real deep
    supervision (auxiliary losses) would require editing the training loop,
    which the fair protocol forbids.  Aggregating the logits of stage-2/3/4
    exits with learnable weights gives the same gradient-shortening effect while
    staying inside the model: every stage-level classifier is trained by the task
    loss and the prediction is their weighted vote.
    """

    def __init__(self, dims: Sequence[int], num_classes: int, p_drop: float = 0.1):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(d, num_classes) for d in dims])
        self.drops = nn.ModuleList([nn.Dropout(p_drop) for _ in dims])
        self.logit_w = nn.Parameter(torch.zeros(len(dims)))     # uniform at init

    def forward(self, feats) -> torch.Tensor:
        w = torch.softmax(self.logit_w, dim=0)
        out = 0.0
        for wi, fc, dp, f in zip(w, self.fcs, self.drops, feats):
            out = out + wi * fc(dp(f))
        return out


class LocalContrastNorm(nn.Module):
    """I7: differentiable local contrast normalisation for CLE.

    Fluorescein intensity in CLE depends on probe pressure and working distance,
    so absolute brightness is not diagnostic while *relative* structure is (the
    Miami criteria grade architecture, not luminance).  A local mean/variance
    normalisation removes that nuisance gain field; the learnable mix is
    zero-initialised so the layer starts as an exact identity.
    """

    def __init__(self, k: int = 9, eps: float = 1e-5, ks=None):
        super().__init__()
        self.eps = eps
        # ks=None -> single window; ks=(5,17) -> multi-scale, each scale gets its
        # own zero-init mix, so the layer still starts as an exact identity.
        self.ks = tuple(ks) if ks else (k,)
        self.pools = nn.ModuleList([
            nn.AvgPool2d(kk, stride=1, padding=kk // 2, count_include_pad=False)
            for kk in self.ks])
        self.mix = nn.Parameter(torch.zeros(len(self.ks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, pool in enumerate(self.pools):
            mu = pool(x)
            var = (pool(x * x) - mu * mu).clamp_min(0.0)
            x_hat = (x - mu) / (var + self.eps).sqrt()
            out = out + torch.tanh(self.mix[i]) * x_hat
        return out


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class CLEConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 3,
                 depths: Sequence[int] = (3, 3, 9, 3), dims: Sequence[int] = (96, 192, 384, 768),
                 drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 use_freq: bool = True, use_aniso: bool = True, use_dual: bool = True,
                 head_type: str = None, aniso_stages: Tuple[int, ...] = (2, 3),
                 head_width: int = 256, head_drop: float = 0.1, freq_mode: str = "sobel",
                 aniso_k: int = 7, aniso_dil: int = 2, head_stage: int = 1,
                 use_fov: bool = False, cov_dim: int = 96, cov_sqrt_iters: int = 5,
                 use_lcn: bool = False, lcn_ks=None, use_fov_in: bool = False,
                 use_sn: bool = False, use_alcn: bool = False, use_asn: bool = False,
                 use_dpf: bool = False):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4
        self.head_type = head_type or ("dual" if use_dual else "linear")
        self.use_dual = self.head_type in ("dual", "res")
        self.aniso_stages = set(aniso_stages)
        self.head_stage = head_stage
        self.fov = FOVMask(size=224) if use_fov else None
        self.lcn = LocalContrastNorm(ks=lcn_ks) if use_lcn else None
        self.fov_in = FOVMask(size=224) if use_fov_in else None
        self.sn = SpectralNotch() if use_sn else None
        self.asn = AngularSpectralNotch() if use_asn else None
        self.dpf = nn.ModuleList([SpectralNotch(), LocalContrastNorm()]) if use_dpf else None
        self.alcn = AdaptiveLCN() if use_alcn else None

        # ---- stem (key path downsample_layers.0.0/.1 preserved) ----
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans * (2 if use_dpf else 1), dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        setattr(stem, "freq", FreqExcite(in_chans, mode=freq_mode) if use_freq else nn.Identity())
        self.downsample_layers.append(stem)

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # ---- stages ----
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks: List[nn.Module] = []
            for j in range(depths[i]):
                if use_aniso and i in self.aniso_stages:
                    blocks.append(AnisoDilatedBlock(
                        dim=dims[i], drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        k=aniso_k, dil=aniso_dil))
                else:
                    blocks.append(Block(
                        dim=dims[i], drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value))
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # ---- head ----
        if self.head_type == "cov":
            self.head = CovPoolHead(dims[-1], dims[-1], num_classes,
                                    dim=cov_dim, sqrt_iters=cov_sqrt_iters)
        elif self.head_type == "mea":
            self.head = MultiExitHead([dims[1], dims[2], dims[3]], num_classes)
        elif self.head_type == "dual":
            self.head = DualPathHead(dims[head_stage], dims[-1], num_classes,
                                     width=head_width, p_drop=head_drop)
        elif self.head_type == "res":
            self.head = ResidualDetailHead(dims[head_stage], dims[-1], num_classes,
                                           p_drop=head_drop)
        else:
            self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # keep head at small init; new branches already zero/neutral
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is getattr(self.head, "fc", None):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.fov is not None:
            return self.fov.masked_avg(x)
        return x.mean([-2, -1])

    def forward_features(self, x: torch.Tensor):
        if self.dpf is not None:
            x = torch.cat([self.dpf[0](x), self.dpf[1](x)], dim=1)
        if self.asn is not None:
            x = self.asn(x)
        if self.sn is not None:
            x = self.sn(x)
        if self.alcn is not None:
            x = self.alcn(x)
        if self.fov_in is not None:
            x = x * self.fov_in.at(x.shape[-2], x.shape[-1])
        if self.lcn is not None:
            x = self.lcn(x)
        pooled = []
        for i in range(4):
            if i == 0:
                x = self.downsample_layers[i].freq(x)
                x = self.downsample_layers[i][1](self.downsample_layers[i][0](x))
            else:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            pooled.append(self._pool(x))              # (B, dims[i]) at every stage
        x4 = x                                        # dense stage-4 map (B,768,7,7)
        pooled[-1] = self.norm(pooled[-1])            # (B, 768) stride-32 semantic
        return pooled, x4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, x4 = self.forward_features(x)
        z_s = pooled[-1]
        if self.head_type == "cov":
            return self.head(x4, z_s)
        if self.head_type == "mea":
            return self.head([pooled[1], pooled[2], pooled[3]])
        if self.use_dual:
            return self.head(pooled[self.head_stage], z_s)
        return self.head(z_s)


# --------------------------------------------------------------------------- #
# Builders (registered names)
# --------------------------------------------------------------------------- #
def _build(num_classes: int, use_freq: bool, use_aniso: bool, use_dual: bool,
           drop_path_rate: float = 0.0, **kw) -> CLEConvNeXt:
    """drop_path_rate defaults to 0 to match the baseline training protocol exactly."""
    return CLEConvNeXt(num_classes=num_classes, depths=(3, 3, 9, 3),
                       dims=(96, 192, 384, 768), drop_path_rate=drop_path_rate,
                       use_freq=use_freq, use_aniso=use_aniso, use_dual=use_dual, **kw)


def cle_convnext_v1(num_classes: int) -> CLEConvNeXt:
    """Full model: I1 (FADES-Sobel) + I2 (ADB stage2-3) + I3 (DPGH stage2+4)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True)


def cle_convnext_nofreq(num_classes: int) -> CLEConvNeXt:
    """Ablation: drop I1."""
    return _build(num_classes, use_freq=False, use_aniso=True, use_dual=True)


def cle_convnext_noaniso(num_classes: int) -> CLEConvNeXt:
    """Ablation: drop I2."""
    return _build(num_classes, use_freq=True, use_aniso=False, use_dual=True)


def cle_convnext_nodual(num_classes: int) -> CLEConvNeXt:
    """Ablation: drop I3."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False)


def cle_convnext_dwt(num_classes: int) -> CLEConvNeXt:
    """I1 variant: Haar-DWT high-frequency energy instead of Sobel/Laplacian."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True, freq_mode="dwt")


def cle_convnext_a3(num_classes: int) -> CLEConvNeXt:
    """I2 variant: anisotropic block only in stage 3 (cheapest)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True, aniso_stages=(3,))


def cle_convnext_a123(num_classes: int) -> CLEConvNeXt:
    """I2 variant: anisotropic block in stages 1-3 (maximal)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True, aniso_stages=(1, 2, 3))


def cle_convnext_k11(num_classes: int) -> CLEConvNeXt:
    """I2 variant: longer strips (k=11) and larger dilation (d=3)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True,
                  aniso_k=11, aniso_dil=3)


def cle_convnext_hres(num_classes: int) -> CLEConvNeXt:
    """Full model v2: FADES + ADB(stage2-3) + residual detail head (I3-v2)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True,
                  head_type="res", head_drop=0.0)


def cle_convnext_hres_a3(num_classes: int) -> CLEConvNeXt:
    """v2 with ADB only in stage 3 (the cheaper, best-scoring I2 setting)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True,
                  head_type="res", head_drop=0.0, aniso_stages=(3,))


def cle_convnext_both(num_classes: int) -> CLEConvNeXt:
    """v2 with Sobel+Haar combined frequency descriptor."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True,
                  head_type="res", head_drop=0.0, freq_mode="both", aniso_stages=(3,))


# ---------- single-innovation variants: needed for the ablation ladder ----------
def cle_convnext_only_freq(num_classes: int) -> CLEConvNeXt:
    """I1 only (no ADB, baseline linear head)."""
    return _build(num_classes, use_freq=True, use_aniso=False, use_dual=False)


def cle_convnext_only_aniso(num_classes: int) -> CLEConvNeXt:
    """I2 only (baseline stem and head, ADB in stage 2-3)."""
    return _build(num_classes, use_freq=False, use_aniso=True, use_dual=False)


def cle_convnext_only_dual(num_classes: int) -> CLEConvNeXt:
    """I3 only, v2 residual head (baseline stem and blocks)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=True,
                  head_type="res", head_drop=0.0)


# ---------- I4: FOV-masked pooling (CLE circular field-of-view prior) ----------
def cle_convnext_fov(num_classes: int) -> CLEConvNeXt:
    """I1(freq) + I2(aniso stage3) + I4(FOV-masked pooling), baseline linear head."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), use_fov=True)


def cle_convnext_fov_hres(num_classes: int) -> CLEConvNeXt:
    """I1 + I2(stage2-3) + I3-v2(residual head) + I4(FOV pooling)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=True,
                  head_type="res", head_drop=0.0, use_fov=True)


def cle_convnext_fov_only(num_classes: int) -> CLEConvNeXt:
    """I4 alone, everything else at baseline."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_fov=True)


# ---------- I5: second-order (covariance) texture head ----------
def cle_convnext_cov(num_classes: int) -> CLEConvNeXt:
    """I1(freq) + I2(aniso stage3) + I5(covariance head)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), head_type="cov")


def cle_convnext_cov_fov(num_classes: int) -> CLEConvNeXt:
    """I1 + I2(stage3) + I4(FOV pool) + I5(covariance head)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), head_type="cov", use_fov=True)


def cle_convnext_cov_only(num_classes: int) -> CLEConvNeXt:
    """I5 alone on the untouched baseline backbone."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  head_type="cov")


def cle_convnext_mea(num_classes: int) -> CLEConvNeXt:
    """I1 + I2(stage3) + I6(multi-exit aggregated head)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), head_type="mea")


def cle_convnext_mea_fov(num_classes: int) -> CLEConvNeXt:
    """I1 + I2(stage3) + I4(FOV pooling) + I6(multi-exit head)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), head_type="mea", use_fov=True)

def cle_convnext_ab(num_classes: int) -> CLEConvNeXt:
    """Two-innovation core: I1(FADES) + I2(ADB in stage 3), baseline linear head.

    This is the clean 2-point model the user asked for: no third head change, so
    any gain is attributable to the frequency prior plus the anisotropic prior.
    """
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,))


def cle_convnext_ab23(num_classes: int) -> CLEConvNeXt:
    """I1 + I2 applied to stages 2-3 (more anisotropy, more parameters)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(2, 3))
def cle_convnext_lcn(num_classes: int) -> CLEConvNeXt:
    """I1 + I2(stage3) + I7(local contrast normalisation)."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), use_lcn=True)


def cle_convnext_lcn_only(num_classes: int) -> CLEConvNeXt:
    """I7 alone on the untouched baseline backbone."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True)
def cle_convnext_lcn_fov(num_classes: int) -> CLEConvNeXt:
    """Two cheap CLE priors only: I7(local contrast) + I4(FOV-masked pooling)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, use_fov=True)


def cle_convnext_lcn_ab(num_classes: int) -> CLEConvNeXt:
    """I7(LCN) + I1(FADES) + I2(ADB stage3): check whether LCN makes the others useful."""
    return _build(num_classes, use_freq=True, use_aniso=True, use_dual=False,
                  aniso_stages=(3,), use_lcn=True)
def cle_convnext_lcn5(num_classes: int) -> CLEConvNeXt:
    """LCN with a 5x5 window (finest local contrast)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, lcn_ks=(5,))


def cle_convnext_lcn13(num_classes: int) -> CLEConvNeXt:
    """LCN with a 13x13 window."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, lcn_ks=(13,))


def cle_convnext_lcn17(num_classes: int) -> CLEConvNeXt:
    """LCN with a 17x17 window (coarse illumination field)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, lcn_ks=(17,))


def cle_convnext_lcn_ms(num_classes: int) -> CLEConvNeXt:
    """Multi-scale LCN: 5x5 + 9x9 + 17x17 with independent zero-init mixes."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, lcn_ks=(5, 9, 17))
def cle_convnext_lcn_fovin(num_classes: int) -> CLEConvNeXt:
    """Two *subtractive* CLE priors: I7(LCN) + input-side FOV disc masking."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, use_fov_in=True)


def cle_convnext_lcn_ms_fovin(num_classes: int) -> CLEConvNeXt:
    """Multi-scale LCN + input-side FOV disc masking."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, lcn_ks=(5, 9, 17), use_fov_in=True)
# ---------- backbone-level redesign: stochastic depth (train-time structural regulariser) ----------
def cle_convnext_lcn_dp10(num_classes: int) -> CLEConvNeXt:
    """LCN + stochastic depth 0.1 on the ConvNeXt backbone."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, drop_path_rate=0.1)


def cle_convnext_lcn_dp20(num_classes: int) -> CLEConvNeXt:
    """LCN + stochastic depth 0.2."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, drop_path_rate=0.2)


def cle_convnext_dp10(num_classes: int) -> CLEConvNeXt:
    """Stochastic depth 0.1 alone (no LCN): isolates the effect of the regulariser."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  drop_path_rate=0.1)
class SpectralNotch(nn.Module):
    """I8: learnable spectral notch for fibre-bundle (honeycomb) artefact removal.

    A probe-based CLE image is formed through a coherent fibre bundle, which
    superimposes a quasi-periodic honeycomb pattern on the tissue signal.  That
    pattern is pure nuisance for grading mucosal architecture, yet it is
    broadband in the spatial domain and therefore hard to remove with local
    filters.  In the Fourier domain it appears as a small set of off-centre
    peaks, so we apply a learnable *radial* mask (one gain per spatial-frequency
    bin, shared across channels) and let the network learn the notch.

    Initialisation: mask = exp(0) = 1 everywhere, i.e. an exact identity, so the
    pretrained backbone sees the untouched image at step 0.
    """

    def __init__(self, size: int = 224, bins: int = 48):
        super().__init__()
        self.bins = bins
        h = w = size
        fy = torch.fft.fftfreq(h).view(-1, 1).expand(h, w // 2 + 1)
        fx = torch.fft.rfftfreq(w).view(1, -1).expand(h, w // 2 + 1)
        r = torch.sqrt(fx ** 2 + fy ** 2) / 0.7071          # normalised radius 0..1
        idx = (r * (bins - 1)).round().long().clamp(0, bins - 1)
        self.register_buffer("bin_idx", idx, persistent=False)
        self.log_gain = nn.Parameter(torch.zeros(bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.bin_idx.shape[0] != h or self.bin_idx.shape[1] != w // 2 + 1:
            fy = torch.fft.fftfreq(h, device=x.device).view(-1, 1).expand(h, w // 2 + 1)
            fx = torch.fft.rfftfreq(w, device=x.device).view(1, -1).expand(h, w // 2 + 1)
            r = torch.sqrt(fx ** 2 + fy ** 2) / 0.7071
            idx = (r * (self.bins - 1)).round().long().clamp(0, self.bins - 1)
        else:
            idx = self.bin_idx
        gain = torch.exp(self.log_gain.clamp(-3.0, 3.0))     # init 1.0
        mask = gain[idx].view(1, 1, h, w // 2 + 1)
        X = torch.fft.rfft2(x, norm="ortho")
        x = torch.fft.irfft2(X * mask, s=(h, w), norm="ortho")
        return x


class AdaptiveLCN(nn.Module):
    """I9: scale-adaptive local contrast normalisation.

    Plain LCN fixes the window k; our ablation showed a sharp optimum at k=9 and
    clear degradation at 5/13/17, i.e. the right scale is image dependent.  Here a
    tiny predictor reads the radial energy spectrum and outputs softmax weights
    over three candidate windows (5/9/17); the normalised responses are mixed by
    those weights.  At init the predictor is zero-biased towards k=9, and each
    response enters through a zero-initialised gain, so the layer starts as an
    identity.
    """

    def __init__(self, ks=(5, 9, 17), bins: int = 24, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.ks = tuple(ks)
        self.bins = bins
        self.pools = nn.ModuleList([
            nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False) for k in self.ks])
        self.spec_fc = nn.Sequential(nn.Linear(bins, 16), nn.GELU(), nn.Linear(16, len(self.ks)))
        nn.init.zeros_(self.spec_fc[-1].weight)
        with torch.no_grad():
            bias = torch.full((len(self.ks),), -4.0)
            bias[1] = 4.0                                    # start dominated by k=9
            self.spec_fc[-1].bias.copy_(bias)
        self.mix = nn.Parameter(torch.zeros(len(self.ks)))

    def _radial_profile(self, x: torch.Tensor) -> torch.Tensor:
        # cheap log-energy over coarse radial bins, computed on |x| via avg-pooled bands
        e = x.abs().mean(dim=1, keepdim=True)
        bands = []
        h, w = e.shape[-2:]
        for i in range(self.bins):
            f = int(round(min(h, w) / (2 ** (1 + (i / self.bins) * 3))))
            f = max(1, min(f, min(h, w)))
            bands.append(F.adaptive_avg_pool2d(e, f).flatten(1).mean(1))
        return torch.stack(bands, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.spec_fc(self._radial_profile(x)), dim=1)   # (B,K)
        out = x
        for i, pool in enumerate(self.pools):
            mu = pool(x)
            var = (pool(x * x) - mu * mu).clamp_min(0.0)
            x_hat = (x - mu) / (var + self.eps).sqrt()
            out = out + (w[:, i].view(-1, 1, 1, 1) * torch.tanh(self.mix[i])) * x_hat
        return out
def cle_convnext_sn(num_classes: int) -> CLEConvNeXt:
    """I8 alone: learnable spectral notch (fibre-bundle artefact suppression)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False, use_sn=True)


def cle_convnext_lcn_sn(num_classes: int) -> CLEConvNeXt:
    """I7(LCN) + I8(spectral notch): the two physics-motivated front-end operators."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_lcn=True, use_sn=True)


def cle_convnext_alcn(num_classes: int) -> CLEConvNeXt:
    """I9 alone: scale-adaptive LCN (predicts the window from the radial spectrum)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False, use_alcn=True)


def cle_convnext_alcn_sn(num_classes: int) -> CLEConvNeXt:
    """I9(adaptive LCN) + I8(spectral notch)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_alcn=True, use_sn=True)
class AngularSpectralNotch(nn.Module):
    """I10: angle-resolved spectral notch (second-generation fibre artefact filter).

    The honeycomb pattern of a fibre bundle is not isotropic: its energy sits on a
    ring at a fixed radius but concentrates at a few *angles*.  A purely radial
    mask (I8) can only attenuate the whole ring, which also removes genuine tissue
    texture at that frequency.  Here the gain is defined on a coarse (radius,
    angle) grid, so the network can notch out the fibre peaks while leaving the
    rest of the ring intact.  Identity initialisation (gain = 1) as in I8.
    """

    def __init__(self, size: int = 224, r_bins: int = 12, a_bins: int = 8):
        super().__init__()
        self.r_bins, self.a_bins = r_bins, a_bins
        self.log_gain = nn.Parameter(torch.zeros(r_bins, a_bins))
        idx = self._build_index(size, size, torch.device("cpu"))
        self.register_buffer("idx", idx, persistent=False)

    @staticmethod
    def _build_index(h: int, w: int, device) -> torch.Tensor:
        fy = torch.fft.fftfreq(h, device=device).view(-1, 1).expand(h, w // 2 + 1)
        fx = torch.fft.rfftfreq(w, device=device).view(1, -1).expand(h, w // 2 + 1)
        r = torch.sqrt(fx ** 2 + fy ** 2)
        theta = torch.atan2(fy, fx) + torch.pi          # 0..2pi
        return torch.stack([r, theta], dim=0)           # (2, H, W/2+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.idx.shape[1] != h or self.idx.shape[2] != w // 2 + 1:
            self.idx = self._build_index(h, w, x.device)
        r, theta = self.idx[0], self.idx[1]
        rmax = r.max().clamp(min=1e-6)
        ri = (r / rmax * (self.r_bins - 1)).round().long().clamp(0, self.r_bins - 1)
        ai = (theta / (2 * torch.pi) * self.a_bins).floor().long().clamp(0, self.a_bins - 1)
        gain = torch.exp(self.log_gain.clamp(-3.0, 3.0))          # init 1.0
        mask = gain[ri, ai].view(1, 1, h, w // 2 + 1)
        X = torch.fft.rfft2(x, norm="ortho")
        return torch.fft.irfft2(X * mask, s=(h, w), norm="ortho")
def cle_convnext_asn(num_classes: int) -> CLEConvNeXt:
    """I10 alone: angle-resolved spectral notch."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False, use_asn=True)


def cle_convnext_asn_sn(num_classes: int) -> CLEConvNeXt:
    """I10 + I8: angle-resolved then radial notch (two-stage fibre suppression)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False,
                  use_sn=True, use_asn=True)
def cle_convnext_dpf(num_classes: int) -> CLEConvNeXt:
    """Dual-prior parallel front-end: SN || LCN -> 6-channel stem (single network)."""
    return _build(num_classes, use_freq=False, use_aniso=False, use_dual=False, use_dpf=True)


def _inflate_stem_for_dpf(state):
    """3->6 input channels: duplicate the stem kernel and halve it (keep activation scale)."""
    k = "downsample_layers.0.0.weight"
    w = state.get(k)
    if w is not None and w.dim() == 4 and w.shape[1] == 3:
        state[k] = torch.cat([w, w], dim=1) / 2.0
    return state
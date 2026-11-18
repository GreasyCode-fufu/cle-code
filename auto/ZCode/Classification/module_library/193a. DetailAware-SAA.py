import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compressed_token_count(num_tokens: int, keep_ratio: float) -> int:
    return max(1, min(num_tokens, int(num_tokens * keep_ratio)))


def _infer_spatial_size(num_tokens: int, spatial_size: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if spatial_size is not None:
        height, width = spatial_size
        if height * width != num_tokens:
            raise ValueError(f"spatial_size {spatial_size} does not match token count {num_tokens}.")
        return height, width

    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    return None


def _detail_score(x: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    B, N, C = x.shape
    score = x.norm(dim=-1)
    hw = _infer_spatial_size(N, spatial_size)
    if hw is not None:
        height, width = hw
        feat = x.transpose(1, 2).reshape(B, C, height, width)
        grad_h = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).abs().mean(dim=1)
        grad_w = (feat[:, :, :, 1:] - feat[:, :, :, :-1]).abs().mean(dim=1)
        grad_h = F.pad(grad_h, (0, 0, 1, 0))
        grad_w = F.pad(grad_w, (1, 0, 0, 0))
        score = score + grad_h.flatten(1) + grad_w.flatten(1)

    score_min = score.amin(dim=1, keepdim=True)
    score_max = score.amax(dim=1, keepdim=True)
    return (score - score_min) / (score_max - score_min + 1e-6)


def restore_feature_norm(source: torch.Tensor, aggregated: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    max_norm = source.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
    agg_norm = aggregated.norm(dim=-1, keepdim=True)
    scaled = aggregated / (agg_norm + eps) * max_norm
    return torch.where(agg_norm > eps, scaled, aggregated)


def density_token_aggregation(x: torch.Tensor, cluster_num: int, subsample_factor: int = 4) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [B, N, C] tokens, got shape {tuple(x.shape)}")

    B, N, C = x.shape
    if N == 0:
        raise ValueError("cannot aggregate an empty token sequence.")

    K = max(1, min(int(cluster_num), N))
    if K == N:
        return x

    device = x.device
    x_norm = F.normalize(x, dim=-1, eps=1e-6)
    sample_count = min(N, max(2 * K, subsample_factor * K))

    samples_per_region = max(1, sample_count // K)
    sub_idx_parts = []
    for idx in range(K):
        start_idx = idx * (N // K)
        end_idx = (idx + 1) * (N // K) if idx < K - 1 else N
        region_size = end_idx - start_idx
        if region_size <= 0:
            continue
        n_samples = min(samples_per_region, region_size)
        region_perm = torch.randperm(region_size, device=device)[:n_samples]
        sub_idx_parts.append(start_idx + region_perm)

    sub_idx = torch.cat(sub_idx_parts) if sub_idx_parts else torch.empty(0, dtype=torch.long, device=device)
    if sub_idx.numel() < sample_count:
        remaining = sample_count - sub_idx.numel()
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        candidates = all_idx[mask]
        if candidates.numel() > 0:
            perm = torch.randperm(candidates.numel(), device=device)[:remaining]
            sub_idx = torch.cat([sub_idx, candidates[perm]])

    sample_count = sub_idx.numel()
    K = min(K, sample_count)
    x_norm_sub = x_norm[:, sub_idx]

    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1.0)
    knn = min(K, sample_count - 1)
    density_sub = torch.topk(sim_sub, k=knn, dim=-1).values.mean(dim=-1)
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    higher_density = density_sub[:, None, :] > density_sub[:, :, None]
    masked_sim_sub = sim_sub.masked_fill(~higher_density, -1e9)
    delta_sub = 1.0 - masked_sim_sub.max(dim=-1).values
    max_density_mask = ~higher_density.any(dim=-1)
    delta_sub[max_density_mask] = 1.0 - sim_sub.min(dim=-1).values[max_density_mask]
    score_sub = density_sub * delta_sub.clamp_min(0.0)

    center_idx = sub_idx[torch.topk(score_sub, k=K, dim=-1).indices]
    centers_norm = torch.gather(x_norm, 1, center_idx[..., None].expand(B, K, x_norm.shape[-1]))
    assign_idx = (x_norm @ centers_norm.transpose(1, 2)).argmax(dim=-1)

    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)
    cluster_counts = one_hot.sum(dim=1).clamp_min(1e-6)
    return torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.unsqueeze(-1)


def detail_aware_aggregation(
    x: torch.Tensor,
    keep_tokens: int,
    detail_ratio: float = 0.35,
    spatial_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Keep high-detail anchors raw, then merge smoother context with DTA."""
    B, N, C = x.shape
    K = max(1, min(int(keep_tokens), N))
    if K == N:
        return x

    detail_budget = int(round(K * detail_ratio))
    detail_budget = max(1, min(detail_budget, K))
    if detail_budget == K:
        score = _detail_score(x, spatial_size)
        detail_idx = score.topk(k=detail_budget, dim=1).indices
        return torch.gather(x, 1, detail_idx[..., None].expand(B, detail_budget, C))

    score = _detail_score(x, spatial_size)
    detail_idx = score.topk(k=detail_budget, dim=1).indices
    detail_tokens = torch.gather(x, 1, detail_idx[..., None].expand(B, detail_budget, C))

    selected = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    selected.scatter_(1, detail_idx, True)
    context_tokens = torch.stack([x[b, ~selected[b]] for b in range(B)], dim=0)
    merged_context = density_token_aggregation(context_tokens, K - detail_budget)

    return torch.cat([detail_tokens, merged_context], dim=1)


class DetailAwareSAA(nn.Module):
    """Detail-aware Selective Aggregation Attention for CV缝合救星 experiments.

    The module follows SAT's full-resolution Query and compressed Key/Value
    design, but reserves a small quota of raw high-frequency/detail tokens before
    clustering the smoother context. This keeps edges and repeated textures more
    visible to every query token while retaining the low-complexity KV path.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        c_ratio: float = 0.5,
        M: float = 0.03,
        detail_ratio: float = 0.35,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}.")

        cr = int(dim * c_ratio)
        if cr <= 0 or cr % num_heads != 0:
            raise ValueError("int(dim * c_ratio) must be positive and divisible by num_heads.")

        self.dim = dim
        self.num_heads = num_heads
        self.cr = cr
        self.scale = qk_scale or (cr // num_heads) ** -0.5
        self.M = float(M)
        self.detail_ratio = float(detail_ratio)

        self.q = nn.Linear(dim, cr, bias=qkv_bias)
        self.k = nn.Linear(dim, cr, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, N, C] tokens, got shape {tuple(x.shape)}")

        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"channel mismatch: expected {self.dim}, got {C}.")

        keep_tokens = _compressed_token_count(N, self.M)
        kv_comp = detail_aware_aggregation(x, keep_tokens, self.detail_ratio, spatial_size)
        kv_comp = restore_feature_norm(x, kv_comp)
        kv_size = kv_comp.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(kv_comp).reshape(B, kv_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv_comp).reshape(B, kv_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return x + self.res_scale * out


DASAA = DetailAwareSAA
SAAPlus = DetailAwareSAA


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = 32, 32

    input_tensor = torch.randn(2, height * width, 128, device=device)
    model = DetailAwareSAA(dim=128, num_heads=8, c_ratio=0.5, M=0.03, detail_ratio=0.35).to(device)
    output_tensor = model(input_tensor, spatial_size=(height, width))

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理!\n")

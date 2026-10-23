import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _compressed_token_count(num_tokens: int, keep_ratio: float) -> int:
    return max(1, min(num_tokens, int(num_tokens * keep_ratio)))


def restore_feature_norm(source: torch.Tensor, aggregated: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Feature Norm Restoration used after token aggregation."""
    max_norm = source.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
    agg_norm = aggregated.norm(dim=-1, keepdim=True)
    scaled = aggregated / (agg_norm + eps) * max_norm
    return torch.where(agg_norm > eps, scaled, aggregated)


def cluster_and_merge(x: torch.Tensor, cluster_num: int, subsample_factor: int = 4) -> torch.Tensor:
    """Density-driven token aggregation for compact Key/Value tokens.

    Args:
        x: Input tokens with shape [B, N, C].
        cluster_num: Number of output representatives.
        subsample_factor: Candidate sampling multiplier for center selection.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [B, N, C] tokens, got shape {tuple(x.shape)}")

    B, N, C = x.shape
    if N == 0:
        raise ValueError("SAA cannot aggregate an empty token sequence.")

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

    if sub_idx_parts:
        sub_idx = torch.cat(sub_idx_parts)
    else:
        sub_idx = torch.empty(0, dtype=torch.long, device=device)

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
    sim_topk_sub = torch.topk(sim_sub, k=knn, dim=-1).values
    density_sub = sim_topk_sub.mean(dim=-1)
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    higher_density = density_sub[:, None, :] > density_sub[:, :, None]
    masked_sim_sub = sim_sub.masked_fill(~higher_density, -1e9)
    max_sim_to_higher = masked_sim_sub.max(dim=-1).values
    delta_sub = 1.0 - max_sim_to_higher

    max_density_mask = ~higher_density.any(dim=-1)
    max_dist_global = 1.0 - sim_sub.min(dim=-1).values
    delta_sub[max_density_mask] = max_dist_global[max_density_mask]
    delta_sub = delta_sub.clamp_min(0.0)

    score_sub = density_sub * delta_sub
    center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1).indices
    center_idx = sub_idx[center_idx_in_sub]

    centers_norm = torch.gather(
        x_norm,
        1,
        center_idx[..., None].expand(B, K, x_norm.shape[-1]),
    )
    sim_token_center = x_norm @ centers_norm.transpose(1, 2)
    assign_idx = sim_token_center.argmax(dim=-1)

    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)
    cluster_counts = one_hot.sum(dim=1).clamp_min(1e-6)
    out = torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.unsqueeze(-1)
    return out


class SAA(nn.Module):
    """Selective Aggregation Attention from SAT for image super-resolution.

    SAA keeps the full-resolution Query tokens and compresses only Key/Value
    tokens by density-driven aggregation, forming efficient global cross-attention.
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

        self.q = nn.Linear(dim, cr, bias=qkv_bias)
        self.k = nn.Linear(dim, cr, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, N, C] tokens, got shape {tuple(x.shape)}")

        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"channel mismatch: expected {self.dim}, got {C}.")

        keep_tokens = _compressed_token_count(N, self.M)
        kv_comp = cluster_and_merge(x, keep_tokens)
        kv_comp = restore_feature_norm(x, kv_comp)
        kv_size = kv_comp.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(kv_comp).reshape(B, kv_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv_comp).reshape(B, kv_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 1024, 128, device=device)
    model = SAA(dim=128, num_heads=8, c_ratio=0.5, M=0.03).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理!\n")

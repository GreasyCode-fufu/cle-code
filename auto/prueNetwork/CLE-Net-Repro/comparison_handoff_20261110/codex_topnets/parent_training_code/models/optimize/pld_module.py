"""CLE-Net v3 新增: PLD — Prototype-guided Local Diagnostic affinity.

错误分析驱动(batch1-4 CM): 所有变体的错误集中于 Benign<->Malignant 互混
(各 12-18 张), Normal 近乎完美。B/M 差异在腺体形态学层面(不规则、筛状、
细胞异型), 需要"与诊断模式比对"的判别机制, 而非更多感受野。

设计: 在 stage3 输出(stride16, 腺体形态层)上:
  1. 维护 P 个可学习诊断原型 (P, C3), 随机小尺度初始化;
  2. 每个空间位置与原型做余弦相似 -> softmax 得亲和分布 (B,P,H,W);
  3. 原型加权和生成"诊断上下文", 与原特征残差(零门控)融合后回注;
  4. 之后正常 downsample 进 stage4。

创新性: 原型引导的判别增强在 ConvNeXt 分类骨干中属新设计; 亲和图
可视化即"模型关注的诊断模式分布", 天然可解释, 契合医学影像论文需求。
工程: 零门控初始化保证初始严格等价; 新增参数约 0.16M。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PLDBridge(nn.Module):
    def __init__(self, dim=384, num_protos=6, proto_scale=0.05):
        super().__init__()
        self.protos = nn.Parameter(proto_scale * torch.randn(num_protos, dim))
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):                       # x: (B, C3, H, W)
        p = F.normalize(self.protos, dim=-1)    # (P, C3)
        xf = F.normalize(x, dim=1)              # (B, C3, H, W)
        sim = torch.einsum("bchw,pc->bphw", xf, p)   # (B, P, H, W)
        aff = sim.softmax(dim=1)
        ctx = torch.einsum("bphw,pc->bchw", aff, p)  # 诊断上下文 (B, C3, H, W)
        return x + self.gate * self.proj(ctx - x.mean(dim=(2, 3), keepdim=True))

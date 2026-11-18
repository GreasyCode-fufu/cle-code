"""R6 冒烟(创新2/3 候选): 严格加载 -> 与 star2 初始逐位等价 -> RNG 配对隔离 -> 梯度/辅助损失 -> 复杂度。
部署口径参数/MACs 不含训练期专用模块(anchor_net / r6edge), 推理时二者不执行。"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model  # noqa: E402
from train_cle_fair import load_pretrained_weights_strict  # noqa: E402

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/mobilenet_v3_large-8738ca79.pth"
dev = torch.device("cuda:0")
REF = "mbv3_r6_star2"
names = sys.argv[1:] or ["mbv3_r6_s2_lsbb", "mbv3_r6_s2_lsside", "mbv3_r6_s2_idcbb",
                         "mbv3_r6_s2_anc1", "mbv3_r6_s2_anc03", "mbv3_r6_s2_edge"]
TRAIN_ONLY = ("anchor_net", "r6edge")


def build(name, seed=0):
    torch.manual_seed(seed)
    m = build_model(name, num_classes=3)
    nxt = torch.rand(1).item()                  # 构建后的全局随机流位置(检验 RNG 隔离)
    load_pretrained_weights_strict(m, PTH, name)
    return m.to(dev), nxt


def deploy_params(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.split(".")[0] not in TRAIN_ONLY)


def macs_of(m):
    from thop import profile
    import copy
    mm = copy.deepcopy(m).cpu().eval()
    macs, _ = profile(mm, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    return macs / 1e9


def cpu_ms(name, iters=30):
    torch.set_num_threads(1)
    m = build_model(name, 3).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(5):
            m(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            m(x)
            ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1000


torch.manual_seed(1)
x = torch.randn(8, 3, 224, 224, device=dev)
y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], device=dev)
ref, nxt_ref = build(REF)
ref.eval()
with torch.no_grad():
    out_ref = ref(x)
p_ref = deploy_params(ref)
print(f"[ref] {REF}: deploy params {p_ref/1e6:.3f}M, MACs {macs_of(ref):.4f}G, cpu1t {cpu_ms(REF):.2f}ms")

for name in names:
    m, nxt = build(name)
    same_head = all(torch.equal(a, b) for a, b in zip(m.classifier.state_dict().values(),
                                                     ref.classifier.state_dict().values()))
    m.eval()
    with torch.no_grad():
        d = (m(x) - out_ref).abs().max().item()
    m.train()
    with torch.amp.autocast("cuda"):
        loss = F.cross_entropy(m(x), y)
        aux = m.mfm.loss if m.mfm is not None else None
        if aux is not None:
            loss = loss + m.mfm_lambda * aux
    loss.backward()
    info = []
    if hasattr(m, "r6agg"):
        info.append(f"agg.bn.γ grad={m.r6agg.bn.weight.grad.abs().sum().item():.2e}")
    if hasattr(m, "anchor_net"):
        n_train = sum(p.requires_grad for p in m.anchor_net.parameters())
        info.append(f"anchor_loss@init={aux.item():.2e} anchor_trainable={n_train}")
        assert aux.item() == 0.0 and n_train == 0
    if hasattr(m, "r6edge"):
        g = sum(p.grad.abs().sum().item() for p in m.r6edge.parameters() if p.grad is not None)
        info.append(f"edge_loss@init={aux.item():.3f} edge_grad={g:.2e}")
        assert aux.item() > 0 and g > 0
    g_head = sum(p.grad.abs().sum().item() for p in m.classifier.parameters() if p.grad is not None)
    m._aux = None                               # 释放带计算图的辅助损失张量(否则 deepcopy 测 MACs 失败)
    pd = deploy_params(m)
    print(f"{name:18s} deploy={pd/1e6:.3f}M (Δ vs star2 {(pd-p_ref)/1e3:+.1f}K) "
          f"all={sum(p.numel() for p in m.parameters())/1e6:.3f}M MACs={macs_of(m):.4f}G "
          f"cpu1t={cpu_ms(name):.2f}ms | same_head_init={same_head} rng_paired={nxt == nxt_ref} "
          f"max|Δlogits|_init={d:.1e} head_grad={g_head:.1e} | " + " ".join(info))
    assert same_head and nxt == nxt_ref and d == 0.0 and g_head > 0
print("[R6 A/C SMOKE PASSED]")

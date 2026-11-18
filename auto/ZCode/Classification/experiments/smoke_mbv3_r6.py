"""R6 冒烟: 构建 -> 严格加载审计 -> 主干逐位等价 -> 前向/反向梯度 -> Params/MACs/CPU 延迟。
用法: python experiments/smoke_mbv3_r6.py [model ...]   (缺省测全部 R6 注册模型)"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model, MODEL_REGISTRY  # noqa: E402
from train_cle_fair import load_pretrained_weights_strict  # noqa: E402
from models.optimize.mbv3_cle_r6 import MBV3R6  # noqa: E402

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/mobilenet_v3_large-8738ca79.pth"
dev = torch.device("cuda:0")
names = sys.argv[1:] or [k for k in MODEL_REGISTRY if k.startswith("mbv3_r6_")]


def cpu_ms(m, iters=30):
    torch.set_num_threads(1)
    m = m.cpu().eval()
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


def macs_of(m):
    from thop import profile
    import copy
    mm = copy.deepcopy(m).cpu().eval()
    macs, _ = profile(mm, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    return macs / 1e9


torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224, device=dev)
y = torch.tensor([0, 1, 2, 1], device=dev)
ref = build_model("mobilenet_v3_large", num_classes=3).to(dev)
load_pretrained_weights_strict(ref, PTH, "mobilenet_v3_large")
ref.eval()
with torch.no_grad():
    f0 = ref.features(x)
    out_ref = ref(x)

# forward_pooled 实现自检: 原头 MBV3R6(head="mlp") 载入 ref 全部权重后输出须逐位一致
chk = MBV3R6(3, head="mlp").to(dev)
chk.load_state_dict(ref.state_dict())
chk.eval()
with torch.no_grad():
    d = (chk(x) - out_ref).abs().max().item()
print(f"[self-check] MBV3R6(head=mlp) vs mobilenet_v3_large max|Δlogits|={d:.2e}")
assert d == 0.0

p_ref = sum(p.numel() for p in ref.parameters())
print(f"{'model':16s} {'params':>8s} {'Δparams':>9s} {'head':>8s} {'MACs(G)':>8s} {'cpu1t(ms)':>9s} "
      f"{'max|Δfeat|':>10s} {'grad_head':>9s} {'grad_bb':>9s}")
print(f"{'mobilenet_v3_large':16s} {p_ref/1e6:8.3f} {'—':>9s} {sum(p.numel() for p in ref.classifier.parameters())/1e6:8.3f} "
      f"{macs_of(ref):8.4f} {cpu_ms(build_model('mobilenet_v3_large', 3)):9.2f}")
for name in names:
    m = build_model(name, num_classes=3).to(dev)
    audit = load_pretrained_weights_strict(m, PTH, name)
    assert not audit["unexpected_keys"]
    m.eval()
    with torch.no_grad():
        dfeat = (m.features(x) - f0).abs().max().item()
    m.train()
    with torch.amp.autocast("cuda"):
        loss = F.cross_entropy(m(x), y)
    loss.backward()
    g_head = sum(p.grad.abs().sum().item() for p in m.classifier.parameters() if p.grad is not None)
    g_bb = m.features[12].block[0][0].weight.grad.abs().sum().item()
    n = sum(p.numel() for p in m.parameters())
    nh = sum(p.numel() for p in m.classifier.parameters())
    print(f"{name:16s} {n/1e6:8.3f} {(n-p_ref)/1e6:+9.3f} {nh/1e6:8.3f} {macs_of(m):8.4f} "
          f"{cpu_ms(build_model(name, 3)):9.2f} {dfeat:10.2e} {g_head:9.2e} {g_bb:9.2e}")
    assert dfeat == 0.0 and g_head > 0 and g_bb > 0
print("[R6 SMOKE PASSED]")

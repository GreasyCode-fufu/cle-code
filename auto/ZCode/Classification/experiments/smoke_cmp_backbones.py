"""10个对比主干的预训练严格加载审计 + 前向。"""
import sys, torch
sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth"
PAIRS = [
    ("resnet50",              f"{PTH}/resnet50-0676ba61.pth"),
    ("resnext50_32x4d",       f"{PTH}/resnext50_32x4d-7cdf4587.pth"),
    ("mobilenet_v2",          f"{PTH}/mobilenet_v2-b0353104.pth"),
    ("mobilenet_v3_large",    f"{PTH}/mobilenet_v3_large-8738ca79.pth"),
    ("shufflenet_v2_x1_0",    f"{PTH}/shufflenetv2_x1-5666bf0f80.pth"),
    ("ghostnet",              f"{PTH}/ghostnet_1x.pth"),
    ("deit_tiny_patch16_224", f"{PTH}/deit_tiny_patch16_224-a1311bcf.pth"),
    ("deit_small_patch16_224",f"{PTH}/deit_small_patch16_224-cd65a155.pth"),
    ("deit_base_patch16_224", f"{PTH}/deit_base_patch16_224-b5f2ef4d.pth"),
    ("poolformer_s24",        f"{PTH}/poolformer_s24.pth.tar"),
]
dev = torch.device("cuda:0")
for name, ck in PAIRS:
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    wrap = "state_dict" in keys[:3] or "model" in keys[:3]
    print(f"[ckpt] {name}: 顶层键 {list(keys)[:3]}{'  <== 有包装>' if wrap else ''}")
    try:
        from train_cle_fair import load_pretrained_weights_strict
        m = build_model(name, num_classes=3).to(dev)
        load_pretrained_weights_strict(m, ck, name)
        with torch.no_grad():
            y = m(torch.randn(2, 3, 224, 224, device=dev))
        print(f"    -> STRICT LOAD OK, out={tuple(y.shape)}")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"    -> FAIL: {msg[:220]}")

"""注入贡献度实测: 在真实测试数据上测量各创新点注入项相对主干特征的 L2 贡献比。
方法: forward hook 在各注入点同时取主干路径输出与注入项, 计算 ||注入项||2 / ||主干||2。
"""
import sys, json, glob
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model

REPO = "/home/login/data/cle/auto/ZCode/Classification"
dev = torch.device("cuda:0")

# 测试集 200 张采样(每类采样, 固定seed)
paths = []
for c, cls in enumerate(["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]):
    fs = sorted(glob.glob(f"{REPO}/datasets/cle_ac_v1/test/{cls}/*"))
    idx = np.random.RandomState(0).choice(len(fs), min(70, len(fs)), replace=False)
    paths += [(f, c) for f in np.array(fs)[idx]]

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

out = {}
for seed in (42, 43, 44, 45):
    m = build_model("convnext_wds_wtb_k3_msf", num_classes=3).to(dev)
    sd = torch.load(f"{REPO}/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed{seed}/convnext_wds_wtb_k3_msf/model_best.pth",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd); m.eval()

    stats = {"wds": [], "wtb_s3": [], "wtb_s4": [], "msf_g3": [], "msf_g2": []}

    # monkey-patch 收集注入比: 复算各分支
    orig_ds0conv = m.downsample_layers[0][0]
    with torch.no_grad():
        for p, y in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            # A: stem
            stem_out = orig_ds0conv(x)
            wds_out = m.wds(x)
            inj = m.wds_gate.view(1,-1,1,1) * wds_out
            stats["wds"].append((inj.norm() / stem_out.norm()).item())
            # B: 各块
            h = m.downsample_layers[0][1](stem_out + inj); h = m.stages[0](h)
            feats = [h]
            for i in range(1, 4):
                h = m._downsample(i, h); h = m.stages[i](h); feats.append(h)
            # B 内部: 逐块测(stage3/4)
            t = feats[2 - 1]  # stage2输出作为stage3输入
            # 重新走 stage3/4 的块以测注入
            cur = m._downsample(2, feats[1])
            for j, blk in enumerate(m.stages[2]):
                a = blk.dwconv(cur)
                inj_b = blk.wtg.view(1,-1,1,1) * blk.wtc(a)
                stats["wtb_s3"].append((inj_b.norm() / a.norm()).item())
                cur = blk(a + inj_b)  # 近似(不含permute后路径, 足够估计量级)
            cur4 = m._downsample(3, cur)
            for j, blk in enumerate(m.stages[3]):
                a = blk.dwconv(cur4)
                inj_b = blk.wtg.view(1,-1,1,1) * blk.wtc(a)
                stats["wtb_s4"].append((inj_b.norm() / a.norm()).item())
                cur4 = blk(a + inj_b)
            # C: msf
            z = feats[3]
            hw = z.shape[-2:]
            i3 = m.msf_g3.view(1,-1,1,1) * m.msf_p3(F.interpolate(feats[2], size=hw, mode="bilinear", align_corners=False))
            i2 = m.msf_g2.view(1,-1,1,1) * m.msf_p2(F.interpolate(feats[1], size=hw, mode="bilinear", align_corners=False))
            stats["msf_g3"].append((i3.norm() / z.norm()).item())
            stats["msf_g2"].append((i2.norm() / z.norm()).item())

    out[f"seed{seed}"] = {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
    print(f"seed{seed}: " + "  ".join(f"{k}={v:.4f}" for k, v in out[f"seed{seed}"].items()))

json.dump(out, open(f"{REPO}/experiments/contribution_analysis.json", "w"), indent=2)
print("[saved] experiments/contribution_analysis.json")

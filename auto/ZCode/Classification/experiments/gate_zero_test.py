"""判决性实验: 冠军网络 门控置零(数学上=同权重vanilla ConvNeXt) vs 门控保留 的 test 表现。
- 若门控置零后成绩回落到 baseline 水平 => 提升来自分支的边界精修(机制真实);
- 若置零后成绩不变 => 提升主要来自训练轨迹差异(初始化彩票), 需调整论文叙事。
同时统计预测翻转数。
"""
import sys, glob
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model

REPO = "/home/login/data/cle/auto/ZCode/Classification"
dev = torch.device("cuda:0")
CLS = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]

paths, labels = [], []
for c, cls in enumerate(CLS):
    for f in sorted(glob.glob(f"{REPO}/datasets/cle_ac_v1/test/{cls}/*")):
        paths.append(f); labels.append(c)
y = np.array(labels)

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

for seed in (42, 43, 44, 45):
    m = build_model("convnext_wds_wtb_k3_msf", num_classes=3).to(dev)
    sd = torch.load(f"{REPO}/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed{seed}/convnext_wds_wtb_k3_msf/model_best.pth",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd); m.eval()
    preds_on, preds_off = [], []
    with torch.no_grad():
        for i in range(0, len(xs), 64):
            b = xs[i:i+64].to(dev)
            preds_on.append(m(b).argmax(1).cpu().numpy())
            m.wds_gate.zero_(); [p.zero_() for p in m.wdd_gate] if hasattr(m, "wdd_gate") else None
            for mod in m.modules():
                if hasattr(mod, "wtg") and isinstance(mod.wtg, torch.Tensor): pass
            # 置零所有门控
            with torch.no_grad():
                m.wds_gate.zero_()
                for blk in list(m.stages[2]) + list(m.stages[3]):
                    blk.wtg.zero_()
                m.msf_g2.zero_(); m.msf_g3.zero_()
            preds_off.append(m(b).argmax(1).cpu().numpy())
            # 恢复
            m.load_state_dict(sd)
    pon, poff = np.concatenate(preds_on), np.concatenate(preds_off)
    f1_on, f1_off = f1_score(y, pon, average="macro"), f1_score(y, poff, average="macro")
    flip = int((pon != poff).sum())
    # 翻转样本的净效果: 置零后正确数变化
    acc_on, acc_off = (pon == y).sum(), (poff == y).sum()
    print(f"seed{seed}: F1 门控开={f1_on:.4f} 门控关={f1_off:.4f} (Δ={(f1_on-f1_off)*100:+.2f}pt)"
          f" | 预测翻转={flip}/{len(y)} | 正确数 开={acc_on} 关={acc_off}")

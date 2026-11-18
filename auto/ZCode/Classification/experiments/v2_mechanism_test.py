"""V2 机制可证伪检验(在真实test上做推理期干预):
 1) WR小波路径移除: 将 wavelet_scale 置零 => WR 退化为继承预训练的 7x7 基卷积
    (网络结构不动, 仅去掉小波贡献) => 若 F1 显著下降, 证明小波路径真实承载判别信息。
 2) CH跨尺度贡献移除: 将 msf2_p2/msf2_p3 投影置零 => 拼接头只看到 stage4 池化向量
    => 若 F1 下降, 证明浅/中层跨尺度证据被分类头真实使用。
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

def run(model, intervene):
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), 64):
            preds.append(model(xs[i:i+64].to(dev)).argmax(1).cpu().numpy())
    return f1_score(y, np.concatenate(preds), average="macro")

for name, seeds in [("convnext_v2bc", (42, 43, 44))]:
    for seed in seeds:
        m = build_model(name, num_classes=3).to(dev)
        sd = torch.load(f"{REPO}/weights_cle_ac_v1/{name}_seed{seed}/{name}/model_best.pth",
                        map_location="cpu", weights_only=True)
        m.load_state_dict(sd); m.eval()
        f_full = run(m, None)
        # 干预1: 小波路径移除
        for mod in m.modules():
            if type(mod).__name__ == "WTConv2d":
                for s in mod.wavelet_scale: s.weight.data.zero_()
        f_nowt = run(m, "nowt")
        m.load_state_dict(sd); m.eval()
        # 干预2: 跨尺度投影移除
        m.msf2_p2.weight.data.zero_(); m.msf2_p3.weight.data.zero_()
        f_noms = run(m, "noms")
        m.load_state_dict(sd); m.eval()
        print(f"{name} seed{seed}: 完整={f_full:.4f}  去小波路径={f_nowt:.4f}(Δ{(f_nowt-f_full)*100:+.2f}pt)"
              f"  去跨尺度={f_noms:.4f}(Δ{(f_noms-f_full)*100:+.2f}pt)")

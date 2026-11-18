"""Gate intervention probe: measure each module's ACTUAL contribution on test set.

For each (model, seed) checkpoint, evaluate test macro-F1 under:
  orig  : gate values as trained
  off   : gates forced to 0     (module fully disabled == plain ConvNeXt-T)
  x10   : gates x10
  x100  : gates x100
  one   : gates = 1.0
"""
import os, sys, json, argparse, torch
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score

ROOT = "/home/login/data/cle/auto/Network/CLE-Net"
sys.path.insert(0, ROOT)

from models.optimize.convnext_cle2 import (
    convnext_egb, convnext_scg, convnext_hsc, convnext_scg_hsc,
    convnext_egb_scg, convnext_egb_scg_hsc)
from models.optimize.convnext_cle5 import convnext_esa, convnext_esa_scg_hsc

FACTORY = {
    "convnext_egb": convnext_egb,
    "convnext_scg": convnext_scg,
    "convnext_hsc": convnext_hsc,
    "convnext_esa": convnext_esa,
    "convnext_egb_scg": convnext_egb_scg,
    "convnext_scg_hsc": convnext_scg_hsc,
    "convnext_egb_scg_hsc": convnext_egb_scg_hsc,
    "convnext_esa_scg_hsc": convnext_esa_scg_hsc,
}

GATE_SUFFIX = ("egb_g", "scg_g", "hsc.g", "esa_g")
# 修复后的 SCG/HSC 不再有标量门, 改用末层零初始化实现初始恒等(ReZero 式)。
# 因此"关闭模块"必须同时把这两类零初始化层清零, 否则 off 只关掉了 EGB/ESA,
# SCG/HSC 仍生效 -> 会低估模块贡献。
ZERO_SUFFIX = ("scg.c3.weight", "scg.c3.bias", "hsc.fc2.weight", "hsc.fc2.bias")


def build_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        ps.append(out.argmax(1).cpu())
        ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return float(f1_score(y, p, average="macro")), float(accuracy_score(y, p))


def set_gates(model, mode):
    """mode=off 关全部模块(标量门 + 零初始化末层), 其余 mode 只缩放标量门。"""
    n = 0
    for name, p in model.named_parameters():
        is_gate = name.endswith(GATE_SUFFIX)
        is_zero = name.endswith(ZERO_SUFFIX)
        if not (is_gate or is_zero):
            continue
        with torch.no_grad():
            if mode == "off":
                p.data.zero_()
                n += 1
            elif mode == "x10" and is_gate:
                p.data.mul_(10.0)
                n += 1
            elif mode == "x100" and is_gate:
                p.data.mul_(100.0)
                n += 1
            elif mode == "one" and is_gate:
                p.data.fill_(1.0)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="convnext_egb,convnext_scg,convnext_hsc,convnext_esa,convnext_egb_scg,convnext_scg_hsc,convnext_egb_scg_hsc,convnext_esa_scg_hsc")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--root", default="weights_cle_ac_v1",
                    help="权重根目录(相对仓库或绝对路径)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_root = os.path.join(ROOT, "datasets", "cle_ac_v1", "test")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = build_transform(224, mean, std)
    ds = datasets.ImageFolder(test_root, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)
    print("test samples:", len(ds), "classes:", ds.classes)

    for mname in args.models.split(","):
        for seed in args.seeds.split(","):
            ck = os.path.join(ROOT, args.root,
                              f"{mname}_seed{seed}", mname, "model_best.pth")
            if not os.path.exists(ck):
                print(f"[skip] {mname} seed{seed}: no ckpt {ck}")
                continue
            sd = torch.load(ck, map_location="cpu")
            for k in ("state_dict", "model"):
                if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
            sd = {k[len("module."):] if k.startswith("module.") else k: v
                  for k, v in sd.items()}
            # 严格校验: 修复后参数名已变, strict=False 会静默漏载 ->
            # 门控停留在零初始化, 会让 off==orig 假成立。
            try:
                probe = FACTORY[mname](num_classes=3)
                probe.load_state_dict(sd, strict=True)
                del probe
            except Exception as e:
                print(f"[load-fail] {mname} seed{seed}: {str(e)[:200]}", flush=True)
                continue
            row = {}
            for mode in ["orig", "off", "x10", "x100", "one"]:
                model = FACTORY[mname](num_classes=3)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                ng = set_gates(model, mode)
                model.to(device)
                f1, acc = evaluate(model, loader, device)
                row[mode] = (round(f1, 4), round(acc, 4), ng)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"{mname:24s} s{seed}  " + "  ".join(
                f"{m}={row[m][0]:.4f}/acc{row[m][1]:.4f}" for m in ["orig", "off", "x10", "x100", "one"]),
                f"  (gates={row['orig'][2]})", flush=True)


if __name__ == "__main__":
    main()

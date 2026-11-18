"""demo/eval_d4.py — D4 推理期对称化在各创新网络上的涨点验证(只读实验).

方法来源: 2026-11-15 CLE_ac 数据剖析 + mobilenet_v3_d4sym 实验(见
/home/login/data/cle/auto/Network/demo/CLE-Net-Repro/resnet34_gnn/README.md):
共聚焦视野各向同性 → 推理期对 D4 二面体群(4×90°旋转×翻转, 8 视图)取
logits 平均, 可在同一 checkpoint 上获得稳定的方差缩减收益(训练不变,
单视 vs D4 双模评估为纯因果配对比较)。

本脚本对 prueNetwork 下 6 个创新网络项目(不含 CLE-Net / CLE-Net-Repro):
  复用各项目自己的 evaluate.py(load_model/build_transform/run_inference/
  compute_metrics, 保证构建/加载/预处理/指标口径与该项目的官方评估一致),
  在各自的 CLE 主域 checkpoint 上做 单视图 vs D4 对称化 双模评估。

不修改任何原始项目文件; 全部输出写入 prueNetwork/demo/results/。

用法(每次一个项目, 独立进程避免 models 包名冲突):
  /home/login/miniconda3/envs/sunfuxiang/bin/python demo/eval_d4.py --project GEM-ConvNeXt
  ... --project all   # 逐项目串行
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PRUE = HERE.parent
ZCODE = Path("/home/login/data/cle/auto/ZCode/Classification")
NETWK = Path("/home/login/data/cle/auto/Network/CLE-Net")
TEST_DIR = "/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test"
CLASS_NAMES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]

# 项目配置: root=项目根(入 sys.path), model=该项目注册名, eval_py=其评估脚本,
# ckpt=checkpoint 模板({s}=seed), seeds=CLE 主域已有种子
PROJECTS = {
    "GEM-ConvNeXt": dict(
        root=PRUE / "GEM-ConvNeXt", model="convnext_v6f", eval_py="evaluate.py",
        ckpt=str(ZCODE / "weights_cle_ac_v1_c/convnext_v6f_seed{s}/convnext_v6f/model_best.pth"),
        seeds=[42, 43, 44]),
    "SF-ConvNeXt": dict(
        root=PRUE / "SF-ConvNeXt", model="convnext_v2bc", eval_py="evaluate.py",
        ckpt=str(ZCODE / "weights_cle_ac_v1_c/convnext_v2bc_seed{s}/convnext_v2bc/model_best.pth"),
        seeds=[42, 43, 44]),
    "DSG-Net": dict(
        root=PRUE / "DSG-Net", model="convnext_tiny_cle_oca_strip", eval_py="evaluate.py",
        ckpt=str(NETWK / "weights_cleac3nets_v1/convnext_tiny_cle_oca_strip_seed{s}/convnext_tiny_cle_oca_strip/model_best.pth"),
        seeds=[42, 43, 44]),
    "DPFusion": dict(
        root=PRUE / "DPFusion", model="cle_convnext_sn", eval_py="evaluate.py",
        ckpt=str(NETWK / "weights_cleac3nets_v1/cle_convnext_sn_seed{s}/cle_convnext_sn/model_best.pth"),
        seeds=[42, 43, 44]),
    "EPIC-MobileNet": dict(
        root=PRUE / "EPIC-MobileNet", model="mbv3_fz16_dce4m", eval_py="evaluate.py",
        ckpt=str(ZCODE / "weights_cle_ac_v1_r6/mbv3_fz16_dce4m_seed{s}/mbv3_fz16_dce4m/model_best.pth"),
        seeds=[42, 43, 44, 45, 46]),
    "EF-ConvNeXt": dict(
        root=PRUE / "EF-ConvNeXt", model="convnext_ef", eval_py="eval/evaluate.py",
        ckpt=str(PRUE / "EF-ConvNeXt/weights/convnext_ef_seed{s}.pth"),
        seeds=[42, 43, 44, 45, 46]),
}

METRICS = ("macro_f1", "accuracy", "weighted_f1", "balanced_accuracy",
           "auc_roc_macro", "macro_sensitivity", "macro_specificity",
           "kappa", "mcc")

# ─── 跨域(泛化)实验配置(2026-11-16): 各项目创新网络相对其 baseline 有涨点的域,
#     checkpoint 均为已有训练产物(ZCode 源仓库 / Network 源仓库 / EF 自带)。───
XD = ZCODE / "datasets"
XD_PROJECTS = {
    "GEM-ConvNeXt@covid": dict(root=PRUE / "GEM-ConvNeXt", model="convnext_v6f",
        eval_py="evaluate.py", test=str(XD / "covid_v1/test"),
        ckpt=str(ZCODE / "weights_xd_covid_r5/convnext_v6f_seed{s}/convnext_v6f/model_best.pth")),
    "GEM-ConvNeXt@cub": dict(root=PRUE / "GEM-ConvNeXt", model="convnext_v6f",
        eval_py="evaluate.py", test=str(XD / "cub_v1/test"),
        ckpt=str(ZCODE / "weights_xd_cub_r5/convnext_v6f_seed{s}/convnext_v6f/model_best.pth")),
    "GEM-ConvNeXt@heel": dict(root=PRUE / "GEM-ConvNeXt", model="convnext_v6f",
        eval_py="evaluate.py", test=str(XD / "heelBone/test"),
        ckpt=str(ZCODE / "weights_xd_heel_r5/convnext_v6f_seed{s}/convnext_v6f/model_best.pth")),
    "GEM-ConvNeXt@tcga": dict(root=PRUE / "GEM-ConvNeXt", model="convnext_v6f",
        eval_py="evaluate.py", test=str(XD / "TCGA/test"),
        ckpt=str(ZCODE / "weights_xd_tcga_r5/convnext_v6f_seed{s}/convnext_v6f/model_best.pth")),
    "EPIC-MobileNet@covid": dict(root=PRUE / "EPIC-MobileNet", model="mbv3_fz16_dce4m",
        eval_py="evaluate.py", test=str(XD / "covid_v1/test"),
        ckpt=str(ZCODE / "weights_xd_covid_r6/mbv3_fz16_dce4m_seed{s}/mbv3_fz16_dce4m/model_best.pth")),
    "EPIC-MobileNet@cub": dict(root=PRUE / "EPIC-MobileNet", model="mbv3_fz16_dce4m",
        eval_py="evaluate.py", test=str(XD / "cub_v1/test"),
        ckpt=str(ZCODE / "weights_xd_cub_r6/mbv3_fz16_dce4m_seed{s}/mbv3_fz16_dce4m/model_best.pth")),
    "EPIC-MobileNet@heel": dict(root=PRUE / "EPIC-MobileNet", model="mbv3_fz16_dce4m",
        eval_py="evaluate.py", test=str(XD / "heelBone/test"),
        ckpt=str(ZCODE / "weights_xd_heel_r6/mbv3_fz16_dce4m_seed{s}/mbv3_fz16_dce4m/model_best.pth")),
    "EPIC-MobileNet@tcga": dict(root=PRUE / "EPIC-MobileNet", model="mbv3_fz16_dce4m",
        eval_py="evaluate.py", test=str(XD / "TCGA/test"),
        ckpt=str(ZCODE / "weights_xd_tcga_r6/mbv3_fz16_dce4m_seed{s}/mbv3_fz16_dce4m/model_best.pth")),
    "DSG-Net@covid": dict(root=PRUE / "DSG-Net", model="convnext_tiny_cle_oca_strip",
        eval_py="evaluate.py", test=str(XD / "covid_v1/test"),
        ckpt=str(NETWK / "weights_dsg_covid_v1/convnext_tiny_cle_oca_strip_seed{s}/convnext_tiny_cle_oca_strip/model_best.pth")),
    "DSG-Net@cub": dict(root=PRUE / "DSG-Net", model="convnext_tiny_cle_oca_strip",
        eval_py="evaluate.py", test=str(XD / "cub_v1/test"),
        ckpt=str(NETWK / "weights_dsg_cub_v1/convnext_tiny_cle_oca_strip_seed{s}/convnext_tiny_cle_oca_strip/model_best.pth")),
    "SF-ConvNeXt@busi": dict(root=PRUE / "SF-ConvNeXt", model="convnext_v2bc",
        eval_py="evaluate.py", test=str(XD / "busi_v1/test"),
        ckpt=str(NETWK / "weights_ext_busi_v1/convnext_v2bc_seed{s}/convnext_v2bc/model_best.pth")),
    "SF-ConvNeXt@covid": dict(root=PRUE / "SF-ConvNeXt", model="convnext_v2bc",
        eval_py="evaluate.py", test=str(XD / "covid_v1/test"),
        ckpt=str(NETWK / "weights_ext_covid_v1/convnext_v2bc_seed{s}/convnext_v2bc/model_best.pth")),
    "SF-ConvNeXt@cub": dict(root=PRUE / "SF-ConvNeXt", model="convnext_v2bc",
        eval_py="evaluate.py", test=str(XD / "cub_v1/test"),
        ckpt=str(NETWK / "weights_ext_cub_v1/convnext_v2bc_seed{s}/convnext_v2bc/model_best.pth")),
    "EF-ConvNeXt@busi": dict(root=PRUE / "EF-ConvNeXt", model="convnext_ef",
        eval_py="eval/evaluate.py", test=str(XD / "busi_v1/test"),
        ckpt=str(ZCODE / "weights_busi_ep/convnext_ef_seed{s}/convnext_ef/model_best.pth")),
    "EF-ConvNeXt@covid": dict(root=PRUE / "EF-ConvNeXt", model="convnext_ef",
        eval_py="eval/evaluate.py", test=str(XD / "covid_v1/test"),
        ckpt=str(ZCODE / "weights_covid_ep/convnext_ef_seed{s}/convnext_ef/model_best.pth")),
    "DPFusion@tcga": dict(root=PRUE / "DPFusion", model="cle_convnext_sn",
        eval_py="evaluate.py", test=str(XD / "TCGA/test"),
        ckpt=str(ZCODE / "weights_tcga_ep/cle_convnext_sn_seed{s}/cle_convnext_sn/model_best.pth")),
}
# 各项目 compute_metrics 键名别名(共享谱系里 accuracy 叫 overall_accuracy)
KEY_ALIAS = {"accuracy": "overall_accuracy"}


class D4Wrap(nn.Module):
    """推理期 D4 对称化包装: 8 视图(4×rot90 × flip) logits 平均。零参数。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        outs = []
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            outs.append(self.inner(xr))
            outs.append(self.inner(torch.flip(xr, dims=[3])))
        return sum(outs) / 8.0


class Flip4Wrap(nn.Module):
    """推理期 V4(翻转群{id,H,V,HV})包装: 4 视图 logits 平均。

    翻转不变性由训练增强(H/V flip p=0.5)在所有域显式声明 → 理论上处处安全;
    与 D4 的差异 = 是否引入未声明的 90° 旋转不变性(仅各向同性模态成立)。
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        views = (x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
                 torch.flip(x, dims=[2, 3]))
        return sum(self.inner(v) for v in views) / 4.0


WRAPS = {"single": None, "d4": D4Wrap, "flip4": Flip4Wrap}


def load_project_eval_module(cfg):
    """以模块方式载入该项目自己的 evaluate.py(项目根先入 sys.path)。"""
    root = str(cfg["root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    eval_path = cfg["root"] / cfg["eval_py"]
    spec = importlib.util.spec_from_file_location(f"proj_eval_{cfg['root'].name}", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True,
                   choices=list(PROJECTS) + list(XD_PROJECTS) + ["all", "all-xd"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--modes", default="single,d4",
                   help="逗号分隔: single,d4,flip4; 结果合并进已有 json")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.project == "all-xd":
        plan = [(n, dict(XD_PROJECTS[n], seeds=[42, 43, 44])) for n in XD_PROJECTS]
    elif args.project in XD_PROJECTS:
        plan = [(args.project, dict(XD_PROJECTS[args.project], seeds=[42, 43, 44]))]
    elif args.project == "all":
        plan = [(n, PROJECTS[n]) for n in PROJECTS]
    else:
        plan = [(args.project, PROJECTS[args.project])]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = HERE / "results"
    out_root.mkdir(exist_ok=True)

    for name, cfg in plan:
        mod = load_project_eval_module(cfg)
        out_dir = out_root / name
        out_dir.mkdir(exist_ok=True)
        test_dir = cfg.get("test", TEST_DIR)
        classes = sorted(d for d in os.listdir(test_dir)
                         if os.path.isdir(os.path.join(test_dir, d)))
        img_paths, y_true = mod.load_test_data(test_dir, classes)
        print(f"[{name}] model={cfg['model']} ncls={len(classes)} "
              f"test_n={len(img_paths)} seeds={cfg['seeds']}")
        for seed in cfg["seeds"]:
            ckpt = cfg["ckpt"].format(s=seed)
            if not os.path.exists(ckpt):
                print(f"  seed{seed}: MISSING ckpt {ckpt} — skip")
                continue
            out_json = out_dir / f"seed{seed}.json"
            row = {"project": name, "model": cfg["model"], "seed": seed,
                   "checkpoint": ckpt, "n_test": len(img_paths),
                   "n_classes": len(classes)}
            if out_json.exists():  # --modes 增量运行时合并已有结果
                row.update(json.loads(out_json.read_text()))
            for mode in [m for m in args.modes.split(",") if m]:
                assert mode in WRAPS, mode
                model = mod.load_model(ckpt, device, cfg["model"], len(classes))
                wrap = WRAPS[mode]
                model = model if wrap is None else wrap(model).to(device).eval()
                transform = mod.build_transform(224, [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                t0 = time.time()
                ret = mod.run_inference(model, img_paths, transform,
                                        args.batch_size, device)
                y_prob = ret[0] if isinstance(ret, tuple) else ret  # 有的项目返回(y_prob, eff)
                y_pred = y_prob.argmax(axis=1)
                m = mod.compute_metrics(y_true, y_pred, y_prob,
                                        len(classes), classes)
                row[mode] = {k: float(m[KEY_ALIAS.get(k, k)]) for k in METRICS}
                cm = m.get("confusion_matrix")
                row[mode + "_confusion"] = cm.tolist() if hasattr(cm, "tolist") else cm
                del model
                torch.cuda.empty_cache()
                print(f"  seed{seed} {mode:>6}: macro_f1={row[mode]['macro_f1']:.4f} "
                      f"({time.time()-t0:.0f}s)")
            for a, b in (("d4", "single"), ("flip4", "single")):
                if a in row and b in row:
                    row[f"diff_{a}"] = {k: row[a][k] - row[b][k] for k in METRICS}
            print(f"  seed{seed}  Δ: " + " ".join(
                f"{k}={row['diff_d4'][k]:+.4f}" if "diff_d4" in row else ""
                for k in ("macro_f1",)) + (
                  f" | flip4: {row['diff_flip4']['macro_f1']:+.4f}"
                  if "diff_flip4" in row else ""))
            with open(out_json, "w") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)
        print(f"[{name}] done → {out_dir}")


if __name__ == "__main__":
    main()

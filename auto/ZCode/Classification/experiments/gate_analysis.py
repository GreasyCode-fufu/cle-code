"""门控激活分析: 各创新点零初始化门控经训练后的范数(证明分支真实启用且幅度可控)。
输出: experiments/gate_analysis.json + 终端摘要
"""
import json, glob, torch

REPO = "/home/login/data/cle/auto/ZCode/Classification"
out = {}
for ck in sorted(glob.glob(f"{REPO}/weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed4*/convnext_wds_wtb_k3_msf/model_best.pth")):
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    seed = ck.split("_seed")[1][0]
    r = {}
    # A: WDS 门控 (96ch)
    g = sd["wds_gate"]
    r["wds_gate_mean_abs"] = float(g.abs().mean())
    r["wds_gate_max_abs"] = float(g.abs().max())
    # B: WTB 门控 (每个升级块的 wtg, stage3 x9块 + stage4 x3块)
    wtg = {k: v for k, v in sd.items() if k.endswith(".wtg")}
    s3 = [v.abs().mean().item() for k, v in wtg.items() if k.startswith("stages.2.")]
    s4 = [v.abs().mean().item() for k, v in wtg.items() if k.startswith("stages.3.")]
    r["wtg_stage3_mean_abs_avg"] = sum(s3) / len(s3)
    r["wtg_stage3_max"] = max(s3)
    r["wtg_stage4_mean_abs_avg"] = sum(s4) / len(s4)
    r["wtg_stage4_max"] = max(s4)
    # C: MSF 门控
    r["msf_g3_mean_abs"] = float(sd["msf_g3"].abs().mean())
    r["msf_g2_mean_abs"] = float(sd["msf_g2"].abs().mean())
    # 对照: 残差内层 layer-scale gamma 的量级(原ConvNeXt训练后)
    gam = [v.abs().mean().item() for k, v in sd.items() if k.endswith(".gamma")]
    r["convnext_gamma_mean_abs_avg"] = sum(gam) / len(gam)
    out[f"seed{seed}"] = r

json.dump(out, open(f"{REPO}/experiments/gate_analysis.json", "w"), indent=2)
print(f"{'branch gate':32s}" + "".join(f"{s:>10s}" for s in out))
keys = ["wds_gate_mean_abs", "wtg_stage3_mean_abs_avg", "wtg_stage4_mean_abs_avg",
        "msf_g3_mean_abs", "msf_g2_mean_abs", "convnext_gamma_mean_abs_avg"]
for k in keys:
    print(f"{k:32s}" + "".join(f"{out[s][k]:10.4f}" for s in out))

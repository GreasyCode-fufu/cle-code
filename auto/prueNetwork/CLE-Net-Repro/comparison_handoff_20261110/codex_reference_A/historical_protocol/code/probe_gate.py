import os, sys, glob, json, re, torch

ROOT = sys.argv[1] if len(sys.argv) > 1 else "weights_cle_ac_v1"
out = {}

def unwrap(sd):
    for k in ("state_dict", "model", "net", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    return sd

for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    ck = sorted(glob.glob(os.path.join(p, "*", "model_best.pth")))
    if not ck:
        continue
    try:
        sd = unwrap(torch.load(ck[0], map_location="cpu"))
    except Exception as e:
        out[d] = {"error": str(e)[:120]}
        continue
    if not isinstance(sd, dict):
        out[d] = {"error": "not a dict"}
        continue
    per_stage = {}
    for k, v in sd.items():
        kk = k
        for pref in ("module.", "model.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        if not re.search(r"(egb_g|scg_g|hsc\.g|esa_g)$", kk):
            continue
        kind = kk.split(".")[-1]
        m = re.match(r"stages\.(\d)\.", kk)
        stage = m.group(1) if m else "-"
        try:
            val = float(v.reshape(-1)[0])
        except Exception:
            continue
        per_stage.setdefault(kind, {}).setdefault(stage, []).append(val)
    summ = {}
    for kind, st in per_stage.items():
        summ[kind] = {
            s: {"n": len(v), "mean": round(sum(v) / len(v), 5),
                "min": round(min(v), 5), "max": round(max(v), 5)}
            for s, v in sorted(st.items())
        }
    out[d] = summ

print(json.dumps(out, indent=1))

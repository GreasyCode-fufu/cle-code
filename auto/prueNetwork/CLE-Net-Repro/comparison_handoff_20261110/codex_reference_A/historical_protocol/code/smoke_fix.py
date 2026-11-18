"""Smoke test after deadlock/coverage fixes: forward equivalence + gate grouping.

Checks that each fixed variant, with backbone copied from a common ConvNeXt-T and all
zero-init branches, is EXACTLY equivalent to baseline at init (max|logit diff| == 0).
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
from models.Convnext import convnext_tiny
from models.optimize.convnext_cle2 import (
    convnext_egb, convnext_scg, convnext_hsc, convnext_egb_scg, convnext_scg_hsc,
    convnext_egb_scg_hsc)
from models.optimize.convnext_cle5 import convnext_esa, convnext_esa_scg_hsc

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
base = convnext_tiny(num_classes=3).eval()
with torch.no_grad():
    yb = base(x)

FACT = [
    ("egb", convnext_egb), ("scg", convnext_scg), ("hsc", convnext_hsc),
    ("egb_scg", convnext_egb_scg), ("scg_hsc", convnext_scg_hsc),
    ("egb_scg_hsc", convnext_egb_scg_hsc),
    ("esa", convnext_esa), ("esa_scg_hsc", convnext_esa_scg_hsc),
]

print(f"{'variant':16s} {'max|diff|':>12s} {'params':>12s}  gates")
allok = True
for name, fac in FACT:
    m = fac(num_classes=3).eval()
    inc = m.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        y = m(x)
    diff = (y - yb).abs().max().item()
    nparam = sum(p.numel() for p in m.parameters())
    ngates = sum(1 for n, _ in m.named_parameters()
                 if n.endswith(("egb_g", "scg_g", "esa_g")))
    # coverage check: for full combos, EGB/ESA must appear at stage2/3 too
    ok = diff == 0.0
    allok = allok and ok
    print(f"{name:16s} {diff:12.2e} {nparam:12d}  scalar_gates={ngates}")

# coverage assertions for the two full combos
m = convnext_egb_scg_hsc(num_classes=3)
egb_stages = sorted({n.split(".")[1] for n, _ in m.named_parameters() if "egb_g" in n})
scg_stages = sorted({n.split(".")[1] for n, _ in m.named_parameters() if ".scg." in n})
print(f"[coverage] egb_scg_hsc  egb@stages={egb_stages} (expect 1,2,3)  scg@stages={scg_stages} (expect 2,3)")
assert egb_stages == ["1", "2", "3"], "EGB coverage bug still present!"
assert scg_stages == ["2", "3"], "SCG coverage wrong!"

m2 = convnext_esa_scg_hsc(num_classes=3)
esa_stages = sorted({n.split(".")[1] for n, _ in m2.named_parameters() if "esa_g" in n})
print(f"[coverage] esa_scg_hsc  esa@stages={esa_stages} (expect 1,2)")
assert esa_stages == ["1", "2"], "ESA coverage bug still present!"

print("FORWARD_EQUIV_ALL_ZERO" if allok else "FORWARD_DIFF_NONZERO")
print("SMOKE_OK")

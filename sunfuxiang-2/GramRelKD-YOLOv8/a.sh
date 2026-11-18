python - <<'PY'
import torch
from ultralytics.nn.modules.block import C2f_DSM

x = torch.randn(2, 160, 40, 40)

module = C2f_DSM(
    c1=160,
    c2=96,
    n=1,
    shortcut=False,
    g=1,
    e=0.5,
    active_ratio=0.5,
)

module.eval()

with torch.no_grad():
    y = module(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Hidden channels:", module.c)
print("Internal blocks:", len(module.m))

for i, block in enumerate(module.m):
    print(
        f"Block {i}: "
        f"active={block.active_channels}, "
        f"cheap={block.cheap_channels}, "
        f"ratio={block.active_ratio}"
    )

assert y.shape == (2, 96, 40, 40)
assert module.c == 48
assert len(module.m) == 1
assert module.m[0].active_channels == 24
assert module.m[0].cheap_channels == 24

print("C2f_DSM independent test passed.")
PY
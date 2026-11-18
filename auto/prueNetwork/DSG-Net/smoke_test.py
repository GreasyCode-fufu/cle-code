"""CPU verification, runnable from any directory without original project imports."""
import argparse, hashlib, inspect, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from model import build_model
from models.Convnext import convnext_tiny

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', type=Path, default=Path(
    '/data/cle/sunfuxiang/classifyML/pth/convnext_tiny_1k_224_ema.pth'),
    help='ImageNet ema checkpoint for strict-load + identity check (skip if missing)')
args = ap.parse_args()
torch.set_num_threads(2)

torch.manual_seed(42)
network = build_model(num_classes=3).eval()
torch.manual_seed(20261030)
inputs = torch.randn(1, 3, 224, 224)
with torch.inference_mode():
    outputs = network(inputs)
assert outputs.shape == (1, 3) and torch.isfinite(outputs).all()
params = sum(p.numel() for p in network.parameters())
assert params == 27861027, params

checkpoint = None
if args.checkpoint and args.checkpoint.exists():
    path = args.checkpoint.resolve(strict=True)
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state['model'] if isinstance(state, dict) and 'model' in state else state
    state = dict(state)
    removed = sorted(k for k in state if k.split('.', 1)[0] in ('head',))
    for k in removed:
        del state[k]
    torch.manual_seed(1)
    base = convnext_tiny(num_classes=3).eval()
    torch.manual_seed(1)
    network = build_model(num_classes=3).eval()
    inc_base = base.load_state_dict(state, strict=False)
    inc_net = network.load_state_dict(state, strict=False)
    ok_missing = sorted(inc_net.missing_keys) == sorted(inc_base.missing_keys +
            ['oca1.scale', 'oca2.scale'] +
            [k for k in inc_net.missing_keys if k.startswith('stages.2.') and '.dose.' in k])
    assert not inc_net.unexpected_keys and ok_missing, inc_net
    torch.manual_seed(20261030)
    with torch.inference_mode():
        y0, y1 = base(inputs), network(inputs)
    assert torch.equal(y0, y1), f"identity broken: {(y0 - y1).abs().max()}"
    checkpoint = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'strict_identity_vs_baseline': True,
                  'missing_expected': ['head.*', 'oca1/oca2.scale', 'stages.2.{2,5,8}.dose.*'],
                  'unexpected': 0}

local_imports = {}
for name, module in list(sys.modules.items()):
    if name == 'model' or name.startswith('models'):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            p.relative_to(ROOT)
            local_imports[name] = str(p)
report = {'output_shape': list(outputs.shape), 'finite_output': True, 'parameters': params,
          'local_code_imported_only_from_copy': True, 'imported_local_modules': local_imports,
          'checkpoint': checkpoint, 'logits': outputs.tolist(), 'device': 'cpu',
          'torch': torch.__version__}
print(json.dumps(report, ensure_ascii=False, indent=2))

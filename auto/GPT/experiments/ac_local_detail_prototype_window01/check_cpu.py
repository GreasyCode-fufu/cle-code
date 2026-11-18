from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import sys
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

S = Path(__file__).resolve().parent
R = S.parents[1]
sys.path[:0] = [str(R), str(R/'experiments/ac_qualification_20261024_window01'),
               str(R/'experiments/ac_structure_20261024_window01'), str(R/'experiments')]
from models.optimize.ac_local_detail_prototype_window01.local_detail import LocalDetailResidual, LocalDetailNet, ARMS, counted_flops
from qualification_model import QualifiedBackbone, counted_flops as base_flops
import run_qualification as q

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
save = lambda n, v: (S/n).write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')


def np_ln(x):
    centered = x-x.mean(1, keepdims=True)
    return centered/np.sqrt((centered**2).mean(1, keepdims=True)+1e-5)


def np_blur(x):
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
    out = np.zeros_like(x)
    kernel = np.outer([1., 2., 1.], [1., 2., 1.])/16
    for a in range(3):
        for b in range(3):
            out += padded[:, :, a:a+x.shape[2], b:b+x.shape[3]]*kernel[a, b]
    return out


def fft_blur(x):
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
    shape = (x.shape[2]+4, x.shape[3]+4)
    kernel = np.outer([1., 2., 1.], [1., 2., 1.])/16
    full = np.fft.irfft2(np.fft.rfft2(padded, s=shape, axes=(-2, -1))*np.fft.rfft2(kernel, s=shape), s=shape, axes=(-2, -1))
    return full[:, :, 2:2+x.shape[2], 2:2+x.shape[3]]


def np_descriptor(block, x):
    z = np_ln(x)
    detail = z-np_blur(z)
    if block.kind == 'highpass': return np_ln(detail)
    energy = np.sqrt(np_blur(detail**2)+1e-6)
    gate = energy/(energy+energy.mean((2, 3), keepdims=True)+1e-6)
    if block.kind != 'rawgate': gate -= gate.mean((2, 3), keepdims=True)
    if block.kind == 'misaligned':
        gate = gate.reshape(*gate.shape[:2], -1)[:, :, block.permutation.numpy()].reshape(gate.shape)
    return np_ln(detail*gate)


def official():
    manifest = json.loads((S/'references/source_manifest.json').read_text())
    for item in manifest['files']:
        p = S/'references'/item['path']; raw = p.read_bytes()
        assert sha(p) == item['sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == item['gitblob']
    src = S/'references/official_blurpool.py'
    license = S/'references/LICENSE.md'
    assert 'NonCommercial' in src.read_text() and '4.0' in license.read_text()
    cls = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'BlurPool')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    forbidden = ['open', 'exec', 'eval', 'compile', '__import__', 'os', 'sys', 'subprocess']
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.Name) and n.id in forbidden for n in ast.walk(forward))
    env = {'F': F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])), str(src), 'exec'), env)
    obj = type('ReviewedBlurOperator', (), {})()
    block = LocalDetailResidual(channels=3, spatial=(4, 5)).double()
    obj.filt_size, obj.stride, obj.pad_off = 3, 1, 0
    obj.filt, obj.pad = block.kernel.clone(), nn.ReplicationPad2d(1)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    a, b = env['forward'](obj, x), block.blur(x)
    probe = torch.randn_like(a)
    ga = torch.autograd.grad((a*probe).sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad((b*probe).sum(), x)[0]
    output_error = float((a-b).abs().max().detach())
    gradient_error = float((ga-gb).abs().max())
    assert output_error == gradient_error == 0
    return dict(passed=True, source_sha256=sha(src), license_sha256=sha(license),
                restricted_AST='BlurPool.forward only; constructed binomial3/replicate/stride1; no thirdparty init/network/train executed',
                output_error=output_error, input_gradient_error=gradient_error,
                full_paper_reproduction=False, license='Adobe2019 CC BY-NC-SA4.0; noncommercial research adaptation')


def main():
    assert os.getuid() == 1000 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
    cfg = json.loads((S/'config.json').read_text())
    assert all(sha(R/path) == digest for kind in ['sources', 'locked'] for path, digest in cfg[kind].items())
    torch.manual_seed(20261025)
    mature = official()
    numeric = {}
    for kind in ['contrast', 'rawgate', 'misaligned', 'highpass']:
        block = LocalDetailResidual(kind, channels=3, spatial=(4, 5)).double()
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        actual = block.descriptor(x)
        error = float(np.abs(actual.detach().numpy()-np_descriptor(block, x.detach().numpy())).max())
        blur_error = float(np.abs(block.blur(x).detach().numpy()-np_blur(x.detach().numpy())).max())
        fft_error = float(np.abs(block.blur(x).detach().numpy()-fft_blur(x.detach().numpy())).max())
        assert error < 1e-9 and blur_error < 1e-12 and fft_error < 1e-12
        assert torch.autograd.gradcheck(block.descriptor, (x[:1],), eps=1e-6, atol=2e-5, rtol=1e-3)
        assert torch.allclose(block.descriptor(x[:1]), actual[:1], atol=1e-12, rtol=0)
        flat = torch.randn(2, 3, 1, 1, dtype=torch.float64).expand(2, 3, 4, 5).clone().requires_grad_()
        value = block.descriptor(flat); grad = torch.autograd.grad(value.square().sum(), flat)[0]
        assert torch.isfinite(value).all() and torch.isfinite(grad).all() and value.abs().max() < 1e-10
        numeric[kind] = dict(numpy_descriptor_error=error, numpy_blur_error=blur_error, FFT_replicate_linear_convolution_error=fft_error,
                             doublegradcheck=True, batch_independent=True, finite_spatial_constant=True, constant_descriptor_zero_with_tolerance=True)
    for h, w in [(1, 1), (1, 7), (7, 1), (5, 9)]:
        for kind in ['contrast', 'rawgate', 'misaligned', 'highpass']:
            block = LocalDetailResidual(kind, channels=3, spatial=(h, w)).double()
            x = torch.randn(2, 3, h, w, dtype=torch.float64, requires_grad=True)
            value = block.descriptor(x); grad = torch.autograd.grad(value.square().sum(), x)[0]
            assert torch.isfinite(value).all() and torch.isfinite(grad).all()
    block = LocalDetailResidual(channels=3, spatial=(4, 5)).double()
    x = torch.randn(1, 3, 4, 5, dtype=torch.float64)
    perm = torch.tensor(np.random.default_rng(20261025).permutation(20))
    rearranged = x.flatten(2)[:, :, perm].reshape_as(x)
    assert torch.equal(x.flatten(2).sort(-1).values, rearranged.flatten(2).sort(-1).values)
    counter = float((block.descriptor(x).flatten(2)[:, :, perm]-block.descriptor(rearranged).flatten(2)).abs().max())
    raw = LocalDetailResidual('rawgate', channels=3, spatial=(4, 5)).double()
    highpass = LocalDetailResidual('highpass', channels=3, spatial=(4, 5)).double()
    gate_survives = float((raw.descriptor(x)-highpass.descriptor(x)).abs().max())
    assert counter > 1e-5 and gate_survives > 1e-5
    updates = {}
    for kind in ['contrast', 'rawgate', 'misaligned', 'highpass']:
        block = LocalDetailResidual(kind)
        x = torch.randn(3, 96, 56, 56); target = torch.randn_like(x)
        optimizer = torch.optim.SGD(block.parameters(), lr=.01); records = []
        assert list(dict(block.named_parameters())) == ['up.weight'] and not block.kernel.requires_grad
        for step in range(2):
            optimizer.zero_grad(); loss = F.mse_loss(block(x), target); loss.backward()
            gradients = {n: float(p.grad.norm()) for n, p in block.named_parameters()}
            assert all(torch.isfinite(p.grad).all() for p in block.parameters()) and gradients['up.weight'] > 0
            records.append(dict(loss=float(loss.detach()), gradients=gradients)); optimizer.step()
        assert records[1]['loss'] < records[0]['loss']; updates[kind] = records
    q.verify(); initializations = []
    for seed in [45, 46, 47]:
        q.core.set_seed(seed); base = QualifiedBackbone('convnext_tiny', q.STUDY/'pretrained').eval()
        image = torch.randn(1, 3, 224, 224)
        with torch.no_grad(): expected = base(image)
        for arm in ['plain', 'mom_only', *ARMS]:
            old = copy.deepcopy(base); ids = {id(p) for p in old.parameters()}
            original = {id(p): p.detach().clone() for p in old.parameters()}; rng = torch.get_rng_state().clone()
            model = LocalDetailNet(old, arm).eval()
            assert torch.equal(rng, torch.get_rng_state()) and ids <= {id(p) for p in model.parameters()}
            assert all(torch.equal(p, original[id(p)]) for p in model.parameters() if id(p) in original)
            output = model(image); assert torch.equal(output, expected)
            F.cross_entropy(output, torch.tensor([1])).backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            params = 27822435+(18528 if arm.startswith('mom_') else 0)+(9216 if arm in ARMS else 0)
            assert sum(p.numel() for p in model.parameters()) == params
            flops = counted_flops(model, image, base_flops)
            extra = 63221760 if arm == 'mom_highpass' else (68640768 if arm in ARMS else 0)
            expected_flops = 8909531136+(202309632 if arm.startswith('mom_') else 0)+extra
            assert flops == expected_flops, (arm, flops, expected_flops)
            if arm in ARMS: assert model.detail.up.weight.grad.norm() > 0
            if arm.startswith('mom_'): assert model.encoder[0][0][0].adapter.weight.grad.norm() > 0
            if arm == 'detailcontrast_only': assert not hasattr(model.encoder[0][0][0], 'adapter')
            initializations.append(dict(seed=seed, arm=arm, parameters=params, counted_flops=flops,
                                       logit_exact=True, RNG_preserved=True, original_parameter_ids_and_values_preserved=True,
                                       finite_backward=True, fixed_filter_Fconv_included=True))
            del model, old, original
        del base
    save('numeric_preflight_cpu.json', dict(passed=True, numeric=numeric, official=mature, toy_updates=updates,
         same_channel_marginals_different_spatial_context_counterexample=counter,
         perchannel_gate_survives_LN_counterexample=gate_survives, public_initializations=initializations,
         CLE_images=0, new_training=0, test_images=0, trainable_detail_parameters='up.weight only; no upstream learned chain',
         logical_FLOPs_exclusions='LN/sqrt/centering/elementwise/padding/memory', novelty_confirmed=False))
    save('complete.json', dict(passed=True, proof_sha256=sha(S/'numeric_preflight_cpu.json'),
                             source_config_sha256=sha(S/'config.json'), overall_research_complete=False))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        save('failure.json', dict(error=repr(error), traceback=traceback.format_exc())); raise

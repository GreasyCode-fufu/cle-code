from pathlib import Path
import ast, copy, hashlib, json, math, os, sys, traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

S = Path(__file__).resolve().parent
R = S.parents[1]
sys.path[:0] = [str(R), str(R/'experiments/ac_qualification_20261024_window01'),
               str(R/'experiments/ac_structure_20261024_window01'), str(R/'experiments')]
from models.optimize.ac_midstage_focal_prototype_window01.midstage_focal import FocalCore, FocalResidual, MidstageFocalNet, ARMS, counted_flops
from qualification_model import QualifiedBackbone, counted_flops as base_flops
import run_qualification as q

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
save = lambda n, v: (S/n).write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')


def np_conv1(x, layer):
    w = layer.weight.detach().numpy()[:, :, 0, 0]
    y = np.einsum('oc,bchw->bohw', w, x)
    return y if layer.bias is None else y+layer.bias.detach().numpy()[None, :, None, None]


def np_dw(x, layer):
    weights = layer.weight.detach().numpy()[:, 0]
    k = weights.shape[-1]
    p = np.pad(x, ((0, 0), (0, 0), (k//2, k//2), (k//2, k//2)), mode='edge')
    out = np.zeros_like(x)
    for a in range(k):
        for b in range(k):
            out += p[:, :, a:a+x.shape[2], b:b+x.shape[3]] * weights[None, :, a, b, None, None]
    return out


def np_gelu(x):
    return .5*x*(1+np.vectorize(math.erf)(x/math.sqrt(2)))


def np_ln(x):
    mean = x.mean(1, keepdims=True)
    return (x-mean)/np.sqrt(((x-mean)**2).mean(1, keepdims=True)+1e-5)


def np_core(core, x):
    query, context, gates = np.split(np_conv1(x, core.f), [core.rank, 2*core.rank], axis=1)
    total = np.zeros_like(context)
    for i, layer in enumerate(core.layers):
        context = np_gelu(np_dw(context, layer))
        total += context*gates[:, i:i+1]
    total += np_gelu(context.mean(2, keepdims=True).mean(3, keepdims=True))*gates[:, 2:3]
    mod = np_conv1(total, core.h)
    if core.kind != 'raw': mod -= mod.mean((2, 3), keepdims=True)
    if core.kind == 'misaligned':
        mod = mod.reshape(*mod.shape[:2], -1)[:, :, core.permutation.numpy()].reshape(mod.shape)
    return query+mod if core.kind == 'additive' else query*mod


def official():
    src = S/'references/official_focalnet.py'
    license = S/'references/LICENSE_FocalNet'
    for p, digest in [(src, '1e665f2a979a4000f8d0846db5ed44a8542ca545'),
                      (license, '9e841e7a26e4eb057b24511e7b92d42b257a80e5')]:
        raw = p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == digest
    assert 'MIT License' in license.read_text() and 'Microsoft Corporation' in license.read_text()
    cls = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'FocalModulation')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.Name) and n.id in ['open','exec','eval','compile','__import__','os','sys','subprocess'] for n in ast.walk(forward))
    env = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])), str(src), 'exec'), env)
    core = FocalCore(3, 'raw', padding_mode='zeros').double()
    obj = type('ReviewedFocalOperator', (), {})()
    obj.f = nn.Linear(3, 9, bias=False).double()
    with torch.no_grad():
        obj.f.weight.copy_(core.f.weight[:, :, 0, 0])
    obj.focal_level = 2
    obj.focal_layers = nn.ModuleList([nn.Sequential(copy.deepcopy(layer), nn.GELU()) for layer in core.layers])
    obj.h = copy.deepcopy(core.h)
    obj.act = nn.GELU()
    obj.normalize_modulator = obj.use_postln_in_modulation = False
    obj.proj = obj.proj_drop = nn.Identity()
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    a = env['forward'](obj, x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    b = core(x); probe = torch.randn_like(b)
    ref_params = [x, *obj.f.parameters(), *obj.h.parameters(), *obj.focal_layers.parameters()]
    own_params = [x, *core.f.parameters(), *core.h.parameters(), *core.layers.parameters()]
    ga = torch.autograd.grad((a*probe).sum(), ref_params, retain_graph=True)
    gb = torch.autograd.grad((b*probe).sum(), own_params)
    errors = [float((u.reshape(-1)-v.reshape(-1)).abs().max()) for u, v in zip(ga, gb)]
    err = float((a-b).abs().max().detach())
    assert err < 1e-12 and max(errors) < 1e-12
    return dict(passed=True, source_sha256=sha(src), license_sha256=sha(license),
                restricted_AST='FocalModulation.forward only; proj/proj_drop Identity; zero padding raw modulation core',
                output_error=err, input_and_all_core_parameter_gradient_error=max(errors),
                candidate_replicate_padding_is_adaptation=True, full_paper_reproduction=False)


def main():
    assert os.getuid() == 1000 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
    cfg = json.loads((S/'config.json').read_text())
    assert all(sha(R/path) == digest for kind in ['sources','locked'] for path,digest in cfg[kind].items())
    torch.manual_seed(20261025)
    numeric = {}
    for kind in ['contrast','raw','misaligned','additive']:
        core = FocalCore(3, kind, spatial=(4, 5)).double()
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        expected = np_core(core, x.detach().numpy()); actual = core(x)
        err = float(np.abs(actual.detach().numpy()-expected).max()); assert err < 1e-12
        assert torch.autograd.gradcheck(core, (x[:1],), eps=1e-6, atol=2e-5, rtol=1e-3)
        assert torch.allclose(core(x[:1]), core(x)[:1], atol=1e-12, rtol=0)
        block = FocalResidual(kind, channels=4, rank=3, spatial=(4, 5)).double()
        v = torch.randn(2, 4, 4, 5, dtype=torch.float64)
        oracle = np_ln(np_core(block.core, np_conv1(np_ln(v.numpy()), block.down)))
        desc_error = float(np.abs(block.descriptor(v).detach().numpy()-oracle).max()); assert desc_error < 1e-9
        flat = torch.randn(2, 3, 1, 1, dtype=torch.float64).expand(2,3,4,5).clone().requires_grad_()
        out = core(flat); grad = torch.autograd.grad(out.square().sum(), flat)[0]
        assert torch.isfinite(out).all() and torch.isfinite(grad).all()
        if kind in ['contrast','misaligned']: assert out.abs().max() < 1e-12
        numeric[kind] = dict(numpy_core_error=err, numpy_descriptor_error=desc_error,
                             doublegradcheck=True, batch_independent=True, finite_constant=True,
                             constant_core_zero=kind in ['contrast','misaligned'])
    for h,w in [(1,1),(1,7),(7,1),(5,9)]:
        for kind in ['contrast','raw','misaligned','additive']:
            core = FocalCore(3, kind, spatial=(h,w)).double()
            x = torch.randn(2,3,h,w,dtype=torch.float64,requires_grad=True)
            out = core(x); grad = torch.autograd.grad(out.square().sum(),x)[0]
            assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    core = FocalCore(3, 'contrast').double()
    x = torch.randn(1,3,4,5,dtype=torch.float64); perm = torch.tensor(np.random.default_rng(20261025).permutation(20))
    rearranged = x.flatten(2)[:,:,perm].reshape_as(x)
    assert torch.equal(x.flatten(2).sort(-1).values,rearranged.flatten(2).sort(-1).values)
    counter = float((core(x).flatten(2)[:,:,perm]-core(rearranged).flatten(2)).abs().max().detach())
    assert counter > 1e-6
    mature = official()
    updates = {}
    for kind in ['contrast','raw','misaligned','additive']:
        block = FocalResidual(kind)
        x = torch.randn(3,192,28,28); target = torch.randn_like(x)
        optimizer = torch.optim.SGD(block.parameters(),lr=.01); records = []
        for step in range(2):
            optimizer.zero_grad();loss = F.mse_loss(block(x),target);loss.backward()
            grads = {n:float(p.grad.norm()) for n,p in block.named_parameters()}
            assert all(torch.isfinite(p.grad).all() for p in block.parameters()) and grads['up.weight']>0
            others = [v for n,v in grads.items() if n!='up.weight']
            assert all(v==0 for v in others) if step==0 else all(v>0 for v in others)
            records.append(dict(loss=float(loss.detach()),gradients=grads));optimizer.step()
        assert records[1]['loss'] < records[0]['loss']; updates[kind] = records
    q.verify(); initializations = []
    for seed in [45,46,47]:
        q.core.set_seed(seed); base = QualifiedBackbone('convnext_tiny',q.STUDY/'pretrained').eval()
        image = torch.randn(1,3,224,224)
        with torch.no_grad(): expected = base(image)
        for arm in ['plain','mom_only',*ARMS]:
            old = copy.deepcopy(base); ids = {id(p) for p in old.parameters()}; rng = torch.get_rng_state().clone()
            model = MidstageFocalNet(old,arm).eval()
            assert torch.equal(rng,torch.get_rng_state()) and ids <= {id(p) for p in model.parameters()}
            output = model(image); assert torch.equal(output,expected)
            F.cross_entropy(output,torch.tensor([1])).backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            params = 27822435+(18528 if arm.startswith('mom_') else 0)+(11832 if arm in ARMS else 0)
            assert sum(p.numel() for p in model.parameters()) == params
            flops = counted_flops(model,image,base_flops)
            expected_flops = 8909531136+(202309632 if arm.startswith('mom_') else 0)+(18552576 if arm in ARMS else 0)
            assert flops == expected_flops,(arm,flops,expected_flops)
            if arm in ARMS:
                assert model.midstage.up.weight.grad.norm()>0
                assert all(p.grad.norm()==0 for n,p in model.midstage.named_parameters() if n!='up.weight')
            if arm.startswith('mom_'):
                assert model.encoder[0][0][0].adapter.weight.grad.norm()>0
            if arm=='focalcontrast_only': assert not hasattr(model.encoder[0][0][0],'adapter')
            initializations.append(dict(seed=seed,arm=arm,parameters=params,counted_flops=flops,
                                       logit_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,
                                       finite_backward=True,zero_up_gradient_chain_passed=arm in ARMS))
            del model,old
        del base
    save('numeric_preflight_cpu.json',dict(passed=True,numeric=numeric,official=mature,toy_updates=updates,
         same_channel_marginals_different_spatial_context_counterexample=counter,
         public_initializations=initializations,CLE_images=0,new_training=0,test_images=0,
         logical_FLOPs_exclusions='LN/GELU/centering/elementwise/padding/memory',novelty_confirmed=False))
    save('complete.json',dict(passed=True,proof_sha256=sha(S/'numeric_preflight_cpu.json'),
                             source_config_sha256=sha(S/'config.json'),overall_research_complete=False))


if __name__=='__main__':
    try: main()
    except Exception as e:
        save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

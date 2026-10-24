"""Isolated new-window A-C research. Test inference deliberately absent."""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ast

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import run_data_study as core
core.STUDY = STUDY
core.WEIGHTS = ROOT / 'weights/ac_structure_20261024_window01'
PYTHON = '/home/login/miniconda3/envs/sunfuxiang/bin/python'
COHORT = ROOT / 'experiments/ac_dataset_v1'
ARMS = ['relational', 'no_neighbor', 'no_depth', 'final_only', 'official4', 'convnext']
SEEDS = [45, 46, 47]
OWN_MODEL = ROOT / 'models/optimize/cle_ac_window01.py'
CPU_PREVIOUS = None


def save(path, value):
    core.save_json(path, value)


def git(message):
    paths = [str(STUDY.relative_to(ROOT)), str(OWN_MODEL.relative_to(ROOT))]
    staged = subprocess.check_output([core.GIT, '-C', str(ROOT), 'diff', '--cached', '--name-only'], text=True).splitlines()
    assert all(p.startswith(paths[0] + '/') or p == paths[1] for p in staged), 'unrelated staged changes; refuse commit'
    subprocess.run([core.GIT, '-C', str(ROOT), 'add', '--', *paths], check=True)
    if subprocess.run([core.GIT, '-C', str(ROOT), 'diff', '--cached', '--quiet']).returncode:
        subprocess.run([core.GIT, '-C', str(ROOT), 'commit', '-m', message], check=True)


def source_hashes():
    names = ['run_window01.py', 'audit_window01.py', 'bootstrap.py', 'PROTOCOL.md', 'MODULE_REVIEW_ZH.md']
    files = {str((STUDY / n).relative_to(ROOT)): core.digest(STUDY / n) for n in names}
    for p in [OWN_MODEL, Path(core.__file__), ROOT / 'experiments/pretrained_representations.py',
              ROOT / 'models/optimize/cle_token_adapter.py']:
        files[str(p.relative_to(ROOT))] = core.digest(p)
    return files


def verify():
    cfg = core.read_json(STUDY / 'config.json')
    assert cfg['source_hashes'] == source_hashes(), 'frozen sources changed'
    for name, sha in cfg['locked_files'].items():
        assert core.digest(ROOT / name) == sha, name
    assert core.digest(STUDY / 'jobs.json') == cfg['jobs_sha256']
    return cfg


def inventory():
    rows = []
    for folder, tag in [(ROOT / 'models', 'models'), (Path('/home/login/data/cle/auto/shared/module_library'), 'module_library')]:
        for p in sorted(folder.rglob('*.py')):
            item = dict(source=tag, path=str(p), sha256=core.digest(p), bytes=p.stat().st_size)
            try:
                t = ast.parse(p.read_text(encoding='utf-8-sig'))
                item.update(classes=[n.name for n in t.body if isinstance(n, ast.ClassDef)],
                            functions=[n.name for n in t.body if isinstance(n, ast.FunctionDef)],
                            imports=[ast.unparse(n) for n in ast.walk(t) if isinstance(n, (ast.Import, ast.ImportFrom))])
            except (SyntaxError, UnicodeError) as e:
                item['parse_error'] = str(e)
            rows.append(item)
    save(STUDY / 'module_inventory.json', rows)
    save(STUDY / 'module_inventory_summary.json', dict(counts=dict(collections.Counter(r['source'] for r in rows)),
        parse_errors=[dict(path=r['path'], error=r['parse_error']) for r in rows if 'parse_error' in r],
        policy='Static only; no execution/import of shared module code; venue titles unverified unless explicitly sourced.'))


def prepare():
    assert os.getuid() == 1000 and ROOT.stat().st_uid == 1000
    if (STUDY / 'config.json').exists():
        verify()
        return
    data = core.read_json(COHORT / 'dataset_config.json')
    manifest = core.read_json(COHORT / 'dataset_manifest.json')
    assert core.digest(COHORT / 'dataset_manifest.json') == data['manifest_sha256']
    cachemeta = core.read_json(ROOT/'.cache_data_study/meta.json')
    assert core.digest(ROOT/'.cache_data_study/resized_uint8.npy') == cachemeta['cache_sha256']
    expected = {'train': (2077, 182), 'val': (427, 39), 'test': (576, 41)}
    indices = data['indices']
    groups = {}
    for split, (n, g) in expected.items():
        rows = [r for r in manifest if r['split'] == split]
        assert len(rows) == n and sorted(r['index'] for r in rows) == sorted(indices[split])
        groups[split] = {r['group_id'] for r in rows}
        assert len(groups[split]) == g and all(r['grade'] in ('A', 'B', 'C') for r in rows)
    assert all(not groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    (STUDY / 'manifest.json').write_bytes((COHORT / 'manifest.json').read_bytes())
    refs = []
    old = ROOT / 'experiments/ac_backbone_recipe'
    assert core.read_json(old / 'independent_audit.json')['passed']
    for seed in SEEDS:
        job = next(j for j in core.read_json(old / 'final_jobs.json') if j['arm'] == 'dino_full' and j['seed'] == seed)
        folder = old / 'runs' / job['id']
        comp = core.read_json(folder / 'complete.json')
        signature = hashlib.sha256((json.dumps(job, sort_keys=True) + core.digest(old / 'config.json') + core.digest(core.__file__)).encode()).hexdigest()
        assert comp['signature'] == signature and core.digest(ROOT / comp['checkpoint']) == comp['checkpoint_sha256']
        logs = list(csv.DictReader((folder / 'train_log.csv').open()))
        assert len(logs) == 60 and comp['best_epoch'] == max(range(60), key=lambda i: float(logs[i]['val_macro_f1']))
        dest = STUDY / 'references' / f's{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['complete.json', 'run_meta.json', 'train_log.csv', 'validation_predictions.csv', 'validation_complete.json']:
            (dest / name).write_bytes((folder / name).read_bytes())
        refs.append(dict(seed=seed, source_job=job['id'], source_signature=signature, checkpoint=comp['checkpoint'],
                         checkpoint_sha256=comp['checkpoint_sha256'], val_macro_f1=comp['best_select_f1'],
                         metadata_sha256={p.name: core.digest(p) for p in dest.iterdir()}))
    jobs = [dict(id=f'window01_{arm}_s{seed}', arm=arm, seed=seed, epochs=60,
                 train=indices['train'], select=indices['val'], score=indices['val']) for arm in ARMS for seed in SEEDS]
    save(STUDY / 'jobs.json', jobs)
    locked = [COHORT / 'dataset_config.json', COHORT / 'dataset_manifest.json', COHORT / 'independent_dataset_audit.json',
              STUDY / 'manifest.json', old / 'config.json', old / 'independent_audit.json',
              ROOT / '.cache_data_study/meta.json', ROOT / '.cache_data_study/resized_uint8.npy',
              ROOT / 'weights/pretrained/dinov2_vits14_pretrain.pth']
    import torchvision, torch, timm
    save(STUDY / 'config.json', dict(identity='ac_structure_20261024_window01', thread_id='01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76',
        created_at=core.timestamp(), baseline='dino_full_cls_patch_mean_fixed', references=refs,
        baseline_selection='Highest existing three-seed development macroF1; no historical test ranking.',
        source_hashes=source_hashes(), locked_files={str(p.relative_to(ROOT)): core.digest(p) for p in locked},
        jobs_sha256=core.digest(STUDY / 'jobs.json'), seeds=SEEDS, arms=ARMS, indices=indices,
        recipe=dict(epochs=60, batch_size=32, encoder_lr=1e-5, head_and_readout_lr=1e-4, weight_decay=.05,
                    clip_norm=1., optimizer='AdamW', schedule='cosine_to_zero', amp='dynamic_fp16', cpu_threads=1,
                    workers=2, convnext_stochastic_depth=.1, augmentation='historical unchanged resize256/crop224'),
        runtime=dict(torch=torch.__version__, torchvision=torchvision.__version__, timm=timm.__version__),
        convnext_pretrained_path='/home/login/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth',
        convnext_pretrained_sha256=core.digest('/home/login/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth'),
        success=dict(test_mean_delta_strictly_greater_than=.015, all_seed_deltas_positive=True, patient_paired_ci_lower_positive=True),
        bootstrap=dict(replicates=20000, seed=20261024, dev_family_comparisons=5, family_percentiles=[.5, 99.5]),
        resources=dict(reserve_gpu_mib=6144, sample_seconds=2, slot_limit='measured throughput and memory/CPU; not legacy fixed6/8'),
        test_inference_enabled=False))
    inventory()
    save(STUDY / 'data_boundary_audit.json', dict(passed=True, expected=expected, patient_intersections_empty=True,
        manifest_sha256=data['manifest_sha256'], physical_cache_rows=3595, allowed_training=2077, allowed_selection=427,
        test_images_loaded=0, warning='Cache physically includes all splits. Guarded getitem permits only the explicitly provided split indices.'))
    git('research: preregister new CLE window baseline, relational depth mechanism and paired controls')


def model_for(seed, arm):
    from models.optimize.cle_ac_window01 import CLEDepthNet, ConvNeXtReference
    core.set_seed(seed)
    import torch
    torch.set_num_threads(1)
    return (ConvNeXtReference(ROOT) if arm == 'convnext' else CLEDepthNet(ROOT, arm)).cuda()


class GuardedDataset:
    def __init__(self, indices, training):
        cfg = core.read_json(STUDY / 'config.json')
        allowed = cfg['indices']['train' if training else 'val']
        assert len(indices) == len(set(indices)) and set(indices) <= set(allowed), 'unauthorized dataset access'
        self.indices = list(indices)
        self.base = core.build_dataset(indices, training)
        assert self.base.array.shape == (3595, 256, 256, 3)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        item = self.base[position]
        assert item[3] == self.indices[position]
        return item


def loader(indices, training, seed):
    import torch
    return torch.utils.data.DataLoader(GuardedDataset(indices, training), batch_size=32, shuffle=training,
        num_workers=2, pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(seed))


def optimizer(model):
    import torch
    enc = list(model.encoder.parameters())
    encids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in encids]
    return torch.optim.AdamW([dict(params=enc, lr=1e-5), dict(params=rest, lr=1e-4)], weight_decay=.05)


def extra_metrics(p, y, indices):
    import numpy as np
    m = core.metrics(p, y)
    recs = core.read_json(STUDY / 'manifest.json')
    by = collections.defaultdict(list)
    pred = p.argmax(1)
    for i, yi, pi in zip(indices, y, pred):
        by[recs[i]['group_id']].append(int(yi == pi))
    m['malignant_recall'] = float(np.mean(pred[y == 1] == 1))
    m['patient_equal_accuracy'] = float(np.mean([np.mean(v) for v in by.values()]))
    return m


def signature(job):
    return hashlib.sha256((json.dumps(job, sort_keys=True) + core.digest(STUDY / 'config.json')).encode()).hexdigest()


def done(job):
    p = STUDY / 'runs' / job['id'] / 'validation_complete.json'
    if not p.exists():
        return False
    x = core.read_json(p)
    assert x['signature'] == signature(job) and core.digest(ROOT / x['checkpoint']) == x['checkpoint_sha256']
    assert core.digest(p.parent / 'validation_predictions.csv') == x['prediction_sha256']
    return True


def worker(job_id):
    import numpy as np
    import torch
    from torch.nn import functional as F
    verify()
    job = next(j for j in core.read_json(STUDY / 'jobs.json') if j['id'] == job_id)
    if done(job):
        return
    out = STUDY / 'runs' / job_id
    out.mkdir(parents=True, exist_ok=True)
    ckdir = core.WEIGHTS / job_id
    ckdir.mkdir(parents=True, exist_ok=True)
    checkpoint = ckdir / 'model_best.pth'
    model = model_for(job['seed'], job['arm'])
    val = loader(job['select'], False, job['seed'])
    if not (out / 'complete.json').exists():
        assert not (out / 'train_log.csv').exists(), 'partial training exists: inspect before retry, do not overwrite'
        train = loader(job['train'], True, job['seed'])
        opt = optimizer(model)
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        scaler = torch.amp.GradScaler('cuda')
        meta = dict(job_id=job_id, arm=job['arm'], seed=job['seed'], signature=signature(job),
            created_at=core.timestamp(), git_commit=subprocess.check_output([core.GIT, '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters()),
            train_count=2077, select_count=427, source_hashes=source_hashes())
        save(out / 'run_meta.json', meta)
        history = []
        best = -1.
        start = time.perf_counter()
        for epoch in range(60):
            torch.cuda.reset_peak_memory_stats()
            model.train()
            loss_total = correct = n = skipped = 0
            seen = []
            wait_total = compute_total = 0.
            epoch_start = time.perf_counter()
            iterator = iter(train)
            while True:
                w = time.perf_counter()
                try:
                    x, y, _, idx = next(iterator)
                except StopIteration:
                    break
                wait_total += time.perf_counter() - w
                seen.extend(idx.tolist())
                c = time.perf_counter()
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                assert torch.isfinite(loss), 'nonfinite train loss'
                previous = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                # Initial dynamic scaler may skip overflowed steps as historical baseline did.
                scaler.step(opt)
                scaler.update()
                skipped += int(scaler.get_scale() < previous)
                loss_total += float(loss.detach()) * len(y)
                correct += int((logits.argmax(1) == y).sum())
                n += len(y)
                torch.cuda.synchronize()
                compute_total += time.perf_counter() - c
            assert n == 2077 and sorted(seen) == sorted(job['train']), 'actual training access mismatch'
            train_seconds = time.perf_counter() - epoch_start
            p, y, index = core.infer(model, val)
            assert index == job['select']
            m = extra_metrics(p, y, index)
            schedule.step()
            if m['macro_f1'] > best:
                best, best_epoch = m['macro_f1'], epoch
                torch.save(model.state_dict(), checkpoint.with_suffix('.tmp'))
                os.replace(checkpoint.with_suffix('.tmp'), checkpoint)
            row = dict(epoch=epoch, train_loss=loss_total/n, train_acc=correct/n, val_macro_f1=m['macro_f1'],
                val_accuracy=m['accuracy'], malignant_recall=m['malignant_recall'], patient_equal_accuracy=m['patient_equal_accuracy'],
                amp_skipped_steps=skipped, loader_wait_seconds=wait_total, compute_seconds=compute_total,
                train_samples_per_second=n/train_seconds, train_seconds=train_seconds,
                peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20, elapsed_seconds=time.perf_counter()-start,
                lr_encoder=opt.param_groups[0]['lr'], lr_head=opt.param_groups[-1]['lr'],
                actual_train_count=n, unique_train_count=len(set(seen)),
                train_index_set_sha256=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest())
            history.append(row)
            core.save_csv(out / 'train_log.csv', history)
            print(json.dumps(dict(job=job_id, **row)), flush=True)
        meta.update(completed_at=core.timestamp(), seconds=time.perf_counter()-start, epochs_completed=60,
                    best_select_f1=best, best_epoch=best_epoch, checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=core.digest(checkpoint))
        save(out / 'complete.json', meta)
    comp = core.read_json(out / 'complete.json')
    assert comp['signature'] == signature(job) and core.digest(ROOT / comp['checkpoint']) == comp['checkpoint_sha256']
    model.load_state_dict(torch.load(ROOT / comp['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
    p, y, indices = core.infer(model, val)
    assert indices == job['select'] and abs(extra_metrics(p, y, indices)['macro_f1'] - comp['best_select_f1']) < 1e-12
    core.save_csv(out / 'validation_predictions.csv', core.prediction_rows(p, y, indices))
    save(out / 'validation_complete.json', dict(signature=signature(job), checkpoint=comp['checkpoint'],
        checkpoint_sha256=comp['checkpoint_sha256'], prediction_sha256=core.digest(out / 'validation_predictions.csv'),
        **extra_metrics(p, y, indices)))


def preflight():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from models.optimize.cle_token_adapter import CLETokenAdapterNet, counted_flops
    cfg = verify()
    x, y, _, _ = next(iter(loader(cfg['indices']['train'][:32], True, 45)))
    x, y = x.cuda(), y.cuda()
    core.set_seed(45)
    original = CLETokenAdapterNet(ROOT, 'full_finetune').cuda().eval()
    original_state = {k: v.cpu().clone() for k, v in original.state_dict().items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        expected = original(x)
    del original
    budgets = []
    shared = None
    for arm in ['baseline', *ARMS]:
        model = model_for(45, arm).eval()
        if arm != 'convnext':
            assert all(torch.equal(model.encoder.state_dict()[k].cpu(), v) for k, v in {k[8:]: v for k, v in original_state.items() if k.startswith('encoder.')}.items())
        if arm in ['baseline', 'relational', 'no_neighbor', 'no_depth', 'final_only']:
            assert torch.equal(model.head.weight.cpu(), original_state['head.weight'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                assert torch.equal(model(x), expected), 'zero residual must exactly match baseline logits'
        if arm in ['relational', 'no_neighbor', 'no_depth', 'final_only']:
            state = {k: v.cpu().clone() for k, v in model.readout.state_dict().items()}
            if shared is None:
                shared = state
            else:
                assert all(torch.equal(v, shared[k]) for k, v in state.items()), 'paired readout initialization'
        opt = optimizer(model)
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        flop = counted_flops(model, x[:1])
        torch.cuda.reset_peak_memory_stats()
        model.train()
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if arm == 'relational' and step == 1:
                assert float(model.readout.down.weight.grad.abs().sum()) > 0
                assert float(model.encoder.patch_embed.proj.weight.grad.abs().sum()) > 0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            scaler.step(opt)
            scaler.update()
        budgets.append(dict(arm=arm, parameters=sum(p.numel() for p in model.parameters()), counted_flops=flop,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20, finite_two_updates=True))
        del model, opt
        torch.cuda.empty_cache()
    # Independent NumPy check of the exact neighborhood relation, including boundaries.
    from models.optimize.cle_ac_window01 import NeighborhoodReadout
    z = np.arange(1, 1+4*4*2, dtype=np.float32).reshape(4,4,2)
    actual = F.avg_pool2d(torch.from_numpy(z).permute(2,0,1)[None], 3,1,1,count_include_pad=False)[0].permute(1,2,0).numpy()
    expected_np = np.array([[z[max(0,i-1):min(4,i+2),max(0,j-1):min(4,j+2)].mean((0,1)) for j in range(4)] for i in range(4)])
    assert np.allclose(actual, expected_np)
    assert len({b['parameters'] for b in budgets if b['arm'] in ['relational','no_neighbor','no_depth','final_only']}) == 1
    try:
        GuardedDataset(cfg['indices']['test'][:1], False)
        raise AssertionError('test guard failed')
    except AssertionError as e:
        assert 'unauthorized dataset access' in str(e)
    for ref in cfg['references']:
        model = model_for(ref['seed'], 'baseline')
        model.load_state_dict(torch.load(ROOT/ref['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        p,y,idx = core.infer(model, loader(cfg['indices']['val'], False, ref['seed']))
        saved = list(csv.DictReader((STUDY/'references'/f's{ref["seed"]}'/'validation_predictions.csv').open()))
        target = np.array([[float(r[f'p{k}']) for k in range(3)] for r in saved])
        assert idx == cfg['indices']['val'] and np.max(np.abs(p-target)) < 1e-6
        assert abs(extra_metrics(p,y,idx)['macro_f1']-ref['val_macro_f1']) < 1e-12
        del model
        torch.cuda.empty_cache()
    save(STUDY/'preflight.json',dict(passed=True, timestamp=core.timestamp(), budgets=budgets,
        checks=['split and test loader denial','original DINO logits exact at initialization','all paired readout states identical',
                'two real AMP backward updates/finite gradients','encoder and readout nonzero gradients',
                'NumPy boundary-aware neighborhood reference','equal readout parameter counts',
                'three historical baselines SHA/signatures and full427 independent probability reproduction'],
        flops_excludes='norm, activation, pooling, pointwise products, softmax, data movement; conv/linear and attention counted2FLOPs/MAC'))
    git('test: pass real-image numerical, loader-boundary and baseline reuse preflight')
    print(json.dumps(budgets), flush=True)


def snapshot():
    global CPU_PREVIOUS
    rows = {line.split()[0]: list(map(int, line.split()[1:9])) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')}
    percentages = {}
    iowait = 0.
    if CPU_PREVIOUS:
        for key, vals in rows.items():
            delta = [b-a for a,b in zip(CPU_PREVIOUS[key], vals)]
            total = sum(delta)
            percentages[key] = 100*(total-delta[3]-delta[4])/total if total else 0.
            if key == 'cpu':
                iowait = 100*delta[4]/total if total else 0.
    CPU_PREVIOUS = rows
    g = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    io = Path('/proc/diskstats').read_text()
    return dict(timestamp=core.timestamp(), gpu_util=float(g[0]), gpu_used_mib=float(g[1]), gpu_free_mib=float(g[2]),
        cpu_percent=percentages.get('cpu',0.), cpu_max_core_percent=max([v for k,v in percentages.items() if k!='cpu'],default=0.),
        cpu_iowait_percent=iowait, ram_used_percent=100*(1-mem['MemAvailable']/mem['MemTotal']))


def append_resource(path, value):
    existed = path.exists()
    with path.open('a', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(value))
        if not existed:
            w.writeheader()
        w.writerow(value)


def bench_worker(job_id):
    import torch
    from torch.nn import functional as F
    arm, phase, index = job_id.split(':')
    model = model_for(45+int(index)%3, arm)
    opt = optimizer(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    cfg=verify()
    data=loader(cfg['indices']['train'],True,45)
    model.train()
    iterator=iter(data)
    waits=0.; samples=0
    for step in range(4):
        t=time.perf_counter(); x,y,_,_=next(iterator); waits+=time.perf_counter()-t
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=F.cross_entropy(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update();torch.cuda.synchronize()
    save(STUDY/'profile'/f'ready_{phase}_{index}.json',dict(pid=os.getpid()))
    gate=STUDY/'profile'/f'gate_{phase}.json'
    deadline=time.monotonic()+120
    while not gate.exists():
        assert time.monotonic()<deadline,'benchmark barrier timed out'
        time.sleep(.05)
    starts=core.read_json(gate)['start_unix']
    while time.time()<starts:
        time.sleep(.01)
    start=time.perf_counter();samples=0;waits=0.
    while time.time()<starts+12:
        t=time.perf_counter()
        try:x,y,_,_=next(iterator)
        except StopIteration:
            iterator=iter(data);x,y,_,_=next(iterator)
        waits+=time.perf_counter()-t
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update();torch.cuda.synchronize();samples+=len(y)
    elapsed=time.perf_counter()-start
    save(STUDY/'profile'/f'bench_{arm}_{index}.json',dict(samples=samples,seconds=elapsed,samples_per_second=samples/elapsed,
         loader_wait_seconds=waits, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))


def profile():
    verify();assert core.read_json(STUDY/'preflight.json')['passed']
    folder=STUDY/'profile';folder.mkdir(exist_ok=True)
    results=[];candidates=[1,2,4]
    previous=0.
    for slots in candidates:
        active=[];start=time.perf_counter();resources=[]
        for k in range(slots):
            arm='relational'
            label=f'{slots*100+k}'
            log=(folder/f'profile_{slots}_{k}.log').open('w')
            p=subprocess.Popen([PYTHON,__file__,'bench-worker','--job',f'{arm}:{slots}:{label}'],cwd=ROOT,env=thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active.append((p,log,folder/f'bench_{arm}_{label}.json'))
        deadline=time.monotonic()+120
        while not all((folder/f'ready_{slots}_{slots*100+k}.json').exists() for k in range(slots)):
            assert all(p.poll() is None for p,_,_ in active),'profile failed before barrier; inspect worker log'
            assert time.monotonic()<deadline,'profile preparation timeout'
            time.sleep(.2)
        save(folder/f'gate_{slots}.json',dict(start_unix=time.time()+.5))
        while any(p.poll() is None for p,_,_ in active):
            s=snapshot();s.update(slots=slots);append_resource(folder/'resource_samples.csv',s);resources.append(s);time.sleep(2)
        for p,l,_ in active:
            l.close();assert p.returncode==0, f'profile worker failed; inspect {folder}'
        measures=[core.read_json(p) for _,_,p in active]
        aggregate=sum(v['samples_per_second'] for v in measures)
        row=dict(slots=slots, aggregate_samples_per_second=aggregate, workers=measures,
            mean_gpu_util=sum(s['gpu_util'] for s in resources)/len(resources),max_gpu_used_mib=max(s['gpu_used_mib'] for s in resources),
            mean_cpu_percent=sum(s['cpu_percent'] for s in resources)/len(resources), wall_seconds=time.perf_counter()-start)
        results.append(row)
        save(folder/'throughput_profile.json',results)
        if slots>=4 and row['mean_gpu_util']<80 and aggregate>previous*1.03 and snapshot()['gpu_free_mib']>12000:
            next_slots=slots+max(2,slots//2)
            if next_slots*max(m['peak_gpu_mib'] for m in measures)<43000 and next_slots*3<=os.cpu_count()*2:
                candidates.append(next_slots)
        previous=aggregate
    # Choose the smallest concurrency within 97% of the best aggregate throughput.
    best=max(r['aggregate_samples_per_second'] for r in results)
    selected=next(r['slots'] for r in results if r['aggregate_samples_per_second']>=.97*best)
    save(folder/'selected.json',dict(initial_slots=selected, tested_slots=[r['slots'] for r in results],
        best_measured_aggregate_samples_per_second=best, selection='smallest within97percent of highest measured throughput',
        training_slots_can_grow=True, no_legacy_fixed_limit=True))
    git('perf: measure single and concurrent CLE training throughput before scheduling')


def thread_env():
    return dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')


def run():
    cfg=verify();assert core.read_json(STUDY/'preflight.json')['passed']
    with (STUDY/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=core.read_json(STUDY/'jobs.json');pending=[j for j in jobs if not done(j)]
        completed_ids={j['id'] for j in jobs if j not in pending}
        running={};failures=[];recent=[];next_sample=0
        selected=core.read_json(STUDY/'profile/selected.json');slots=selected['initial_slots']
        # Derived from real preflight footprint with context/optimizer margin, not fixed6/8.
        footprints={r['arm']:max(2048,r['peak_gpu_mib']*1.3+700) for r in core.read_json(STUDY/'preflight.json')['budgets']}
        memory_per_job=min(footprints.values())
        capacity=snapshot()['gpu_free_mib']-6144
        memory_slots=max(1,int(capacity/memory_per_job))
        cpu_slots=max(1,os.cpu_count()//2)
        ceiling=min(len(jobs),memory_slots,cpu_slots)
        slots=min(slots,ceiling)
        logs=STUDY/'logs';logs.mkdir(exist_ok=True)
        save(STUDY/'pipeline_pid.json',dict(pid=os.getpid(),started_at=core.timestamp(),identity=cfg['identity']))
        while pending or running:
            completed=[]
            for pid,(p,j,l) in list(running.items()):
                if p.poll() is None:
                    continue
                l.close();del running[pid]
                if p.returncode or not done(j):
                    failures.append(dict(job=j['id'],returncode=p.returncode,log=str(logs/(j['id']+'.log'))))
                else:
                    completed.append(j['id'])
                    completed_ids.add(j['id'])
            if time.monotonic()>=next_sample:
                s=snapshot();recent.append(s);recent=recent[-15:]
                append_resource(STUDY/'resource_samples.csv',dict(**s,running=len(running),pending=len(pending),slots=slots))
                if len(recent)==15 and pending and len(running)>=slots:
                    if sum(r['gpu_util'] for r in recent)/15<80 and s['cpu_percent']<75 and s['gpu_free_mib']>memory_per_job+6144:
                        slots=min(slots+1,ceiling)
                next_sample=time.monotonic()+2
            while pending and len(running)<slots:
                needed=footprints[pending[0]['arm']]
                reserved=sum(footprints[j['arm']] for p,j,l in running.values())
                if s['gpu_free_mib']<needed+6144 or reserved+needed>capacity or s['cpu_percent']>92:
                    break
                j=pending.pop(0);l=(logs/(j['id']+'.log')).open('a',buffering=1)
                p=subprocess.Popen([PYTHON,__file__,'worker','--job',j['id']],cwd=ROOT,env=thread_env(),stdin=subprocess.DEVNULL,stdout=l,stderr=subprocess.STDOUT)
                running[p.pid]=(p,j,l)
                print('[launch]',j['id'],p.pid,flush=True)
            if completed:
                git('results: retain new CLE window completed jobs '+','.join(completed))
            save(STUDY/'status.json',dict(phase='training' if pending or running else 'needs_independent_audit',
                completed=len(completed_ids),total=len(jobs),pending=len(pending),
                running=[dict(job=j['id'],pid=p.pid) for p,j,l in running.values()],failures=failures,slots=slots,resource=s,updated_at=core.timestamp()))
            if pending or running:
                time.sleep(.5)
        assert not failures,'Some workers failed. Inspect exact logs and own live processes; completed training must only rerun inference.'
        subprocess.run([PYTHON,str(STUDY/'audit_window01.py')],cwd=ROOT,env=thread_env(),check=True)
        save(STUDY/'status.json',dict(phase='round1_audited_continue_development',completed=len(jobs),total=len(jobs),
             updated_at=core.timestamp(),test_used=False,next_action='Analyze train/val evidence and preregister a distinct next mechanism; do not stop the overall goal.'))
        git('results: independently audit first new-window CLE round, preserve all outcomes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','profile','bench-worker','worker','run']);p.add_argument('--job')
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='preflight':preflight()
    elif a.command=='profile':profile()
    elif a.command=='bench-worker':bench_worker(a.job)
    elif a.command=='worker':worker(a.job)
    else:run()

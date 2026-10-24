"""Preregistered new-seed LS replication and equal-capacity/kernel-law controls."""
import argparse
import collections
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import run_ac_architecture as previous
from models.optimize.cle_ls_mechanisms import LSStudyResNet, KernelLawLS
from models.optimize.cle_morphology import PlainControl, LSReference
core = previous.core
COHORT = previous.COHORT
STUDY = ROOT / 'experiments/ac_ls_mechanisms'
core.STUDY = STUDY
core.WEIGHTS = ROOT / 'weights/ac_ls_mechanisms'
ARMS = ['plain', 'conv_ls_budget', 'ls_signed', 'ls_positive', 'ls_global']
SEEDS = [45, 46, 47]


def sources():
    return {name: core.digest(ROOT / path) for name, path in dict(driver='experiments/run_ac_ls_mechanisms.py',
        model='models/optimize/cle_ls_mechanisms.py', reference='models/optimize/cle_morphology.py',
        trainer='experiments/run_data_study.py', cohort='experiments/run_ac_dataset.py',
        previous_driver='experiments/run_ac_architecture.py', protocol='experiments/AC_LS_MECHANISM_PROTOCOL.md').items()}


def commit(message):
    subprocess.run([core.GIT, '-C', str(ROOT), 'add', str(STUDY.relative_to(ROOT)), 'experiments/run_ac_ls_mechanisms.py',
                    'models/optimize/cle_ls_mechanisms.py', 'experiments/AC_LS_MECHANISM_PROTOCOL.md', 'experiments/RESEARCH_LEDGER.md', '.gitignore'], check=True)
    if subprocess.run([core.GIT, '-C', str(ROOT), 'diff', '--cached', '--quiet']).returncode:
        subprocess.run([core.GIT, '-C', str(ROOT), 'commit', '-m', message], check=True)


def verify():
    cfg = core.read_json(STUDY / 'config.json')
    assert cfg['sources'] == sources()
    for path, key in [(COHORT / 'dataset_config.json', 'cohort_sha256'), (STUDY / 'manifest.json', 'manifest_sha256'),
                      (STUDY / 'final_jobs.json', 'jobs_sha256'), (previous.STUDY / 'independent_audit.json', 'round1_audit_sha256')]:
        assert core.digest(path) == cfg[key]
    assert core.digest(core.PRETRAINED) == cfg['pretrained_sha256']
    return cfg


def model_for(seed, arm, width):
    import torch
    model = LSStudyResNet(previous.cohort.ORIGINAL_MODEL(seed), arm, width).cuda()
    torch.set_num_threads(1)
    return model


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY / 'config.json').exists():
        verify()
        return
    assert core.read_json(previous.STUDY / 'independent_audit.json')['passed']
    data = core.read_json(COHORT / 'dataset_config.json')
    indices = data['indices']
    (STUDY / 'manifest.json').write_bytes((COHORT / 'manifest.json').read_bytes())
    target = sum(p.numel() for p in LSReference(256).parameters())
    widths = {w: sum(p.numel() for p in PlainControl(256, w).parameters()) for w in range(32, 257, 8)}
    width = min(widths, key=lambda w: abs(widths[w] - target))
    jobs = [dict(id=f'acls_{arm}_s{seed}', phase='architecture', arm=arm, variant=arm, seed=seed, epochs=60,
                 train=indices['train'], select=indices['val'], score=indices['val'], sample_weights={}, control_width=width)
            for arm in ARMS for seed in SEEDS]
    core.save_json(STUDY / 'final_jobs.json', jobs)
    core.save_json(STUDY / 'config.json', dict(sources=sources(), cohort_sha256=core.digest(COHORT / 'dataset_config.json'),
        round1_audit_sha256=core.digest(previous.STUDY / 'independent_audit.json'), manifest_sha256=core.digest(STUDY / 'manifest.json'),
        jobs_sha256=core.digest(STUDY / 'final_jobs.json'), pretrained_sha256=core.digest(core.PRETRAINED), arms=ARMS, seeds=SEEDS,
        control_width=width, ls_modifier_parameters=target, conv_modifier_parameters=widths[width],
        control_relative_parameter_error=abs(widths[width] - target) / target,
        new_trainings=15, initial_parallel=6, max_parallel=8, reserve_gpu_mib=6144, new_job_gpu_mib=2048,
        primary='ls_signed', primary_references=['plain', 'conv_ls_budget'], mechanism_references=['ls_positive', 'ls_global'],
        primary_minimum_delta=.005, family_percentiles=[.625, 99.375], bootstrap_replicates=20000, bootstrap_rng=20261024))
    commit('research: freeze new-seed AC LS replication and capacity/kernel controls')


def done(job):
    path = STUDY / 'runs' / job['id'] / 'validation_complete.json'
    if not path.exists():
        return False
    value = core.read_json(path)
    assert core.complete(job) and value['signature'] == core.signature(job) and value['n'] == 427
    for name, sha in value['file_hashes'].items():
        assert core.digest(path.parent / name) == sha
    return True


def worker(job_id):
    import torch
    verify()
    job = next(j for j in core.read_json(STUDY / 'final_jobs.json') if j['id'] == job_id)
    if done(job):
        return
    core.new_model = lambda seed: model_for(seed, job['arm'], job['control_width'])
    if not core.complete(job):
        core.run_worker(job_id)
    out = STUDY / 'runs' / job_id
    comp = core.read_json(out / 'complete.json')
    model = core.new_model(job['seed'])
    model.load_state_dict(torch.load(ROOT / comp['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    change = []
    def hook(module, inputs, output):
        value = (output - inputs[0]).abs().flatten(1).mean(1) / (inputs[0].abs().flatten(1).mean(1) + 1e-8)
        change.extend(value.detach().cpu().float().tolist())
    h = model.modifier.register_forward_hook(hook)
    p, y, idx = core.infer(model, core.loader(job['score'], False, seed=job['seed']))
    h.remove()
    assert abs(core.metrics(p, y)['macro_f1'] - comp['best_select_f1']) < 1e-12
    core.save_csv(out / 'validation_predictions.csv', core.prediction_rows(p, y, idx))
    model.modifier.enabled = False
    off, yo, io = core.infer(model, core.loader(job['score'], False, seed=job['seed']))
    assert idx == io and np.array_equal(y, yo)
    core.save_csv(out / 'off_predictions.csv', core.prediction_rows(off, yo, io))
    core.save_json(out / 'validation_complete.json', dict(signature=core.signature(job), checkpoint_sha256=comp['checkpoint_sha256'],
        arm=job['arm'], seed=job['seed'], n=len(y), parameters=sum(p.numel() for p in model.parameters()),
        modifier_parameters=sum(p.numel() for p in model.modifier.parameters()),
        diagnostics=dict(stream_change_mean=float(np.mean(change)), off_prediction_disagreement=float(np.mean(p.argmax(1) != off.argmax(1))),
                         off_metrics=core.metrics(off, yo)),
        file_hashes={name: core.digest(out / name) for name in ['validation_predictions.csv', 'off_predictions.csv']}, **core.metrics(p, y)))


def check():
    import torch
    import torch.nn.functional as F
    from models.optimize.cle_token_adapter import counted_flops
    cfg = verify()
    data = core.read_json(COHORT / 'dataset_config.json')
    x, y, _, _ = next(iter(core.loader(data['indices']['train'][:32], False, seed=45)))
    x, y = x.cuda(), y.cuda()
    plain = previous.cohort.new_model(45).eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        reference = plain(x)
    initial = {k: v.detach().cpu().clone() for k, v in plain.state_dict().items()}
    del plain
    old = previous.model_for(45, 'ls_reference', cfg['control_width']).eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        ls_reference = old(x)
    ls_initial = {k: v.detach().cpu().clone() for k, v in old.modifier.branch.state_dict().items()}
    del old
    budgets = []
    for arm in ARMS:
        model = model_for(45, arm, cfg['control_width']).eval()
        assert all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.backbone.state_dict().items())
        model.modifier.enabled = False
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            assert torch.equal(model(x), reference)
        model.modifier.enabled = True
        if arm.startswith('ls_'):
            branch = model.modifier.branch
            assert all(torch.equal(v.detach().cpu(), ls_initial[k]) for k, v in branch.state_dict().items())
            if arm == 'ls_signed':
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    assert torch.equal(model(x), ls_reference)
            with torch.no_grad():
                weights = branch.weights(torch.randn(2, 256, 5, 5, device='cuda'))
                if arm == 'ls_positive':
                    assert (weights >= 0).all() and torch.allclose(weights.sum(2), torch.ones_like(weights[:, :, 0]), atol=2e-6)
                if arm == 'ls_global':
                    assert torch.equal(weights, weights[:, :, :, :1, :1].expand_as(weights))
        flops = counted_flops(model, x[:1])
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        params = list(model.modifier.parameters()) if arm != 'plain' else list(model.backbone.fc.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        before = [p.detach().clone() for p in params]
        scaler.step(optimizer)
        scaler.update()
        assert any(not torch.equal(p, value) for p, value in zip(params, before))
        budgets.append(dict(arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                            modifier_parameters=sum(p.numel() for p in model.modifier.parameters()),
                            counted_flops=flops, peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20))
        del model, optimizer, params, before
        torch.cuda.empty_cache()
    core.save_json(STUDY / 'preflight_validation.json', dict(passed=True, timestamp=core.timestamp(), budgets=budgets,
        checks=['paired backbone/head and branch initial states', 'disabled branch equals plain AMP logits',
                'signed LS equals immutable round1 reference AMP output', 'positive/global kernel laws',
                'real batch32 finite AMP gradients/updates', 'parameter/FLOP/peak memory budgets']))
    commit('test: validate AC LS kernel-law controls and exact signed-reference equivalence')
    print(json.dumps(budgets), flush=True)


def run():
    with (STUDY / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = verify()
        assert core.read_json(STUDY / 'preflight_validation.json')['passed']
        jobs = core.read_json(STUDY / 'final_jobs.json')
        pending = [j for j in jobs if not done(j)]
        running = {}
        retries = collections.Counter()
        logs = STUDY / 'logs'
        logs.mkdir(exist_ok=True)
        slots = cfg['initial_parallel']
        start = time.monotonic()
        next_sample = 0
        recent = []
        while pending or running:
            completed = []
            for pid, (p, job, f) in list(running.items()):
                if p.poll() is None:
                    continue
                f.close()
                del running[pid]
                if p.returncode or not done(job):
                    retries[job['id']] += 1
                    if retries[job['id']] <= 1:
                        pending.append(job)
                    else:
                        raise RuntimeError(job['id'] + ' failed twice; inspect logs and live children')
                else:
                    completed.append(job['id'])
            if time.monotonic() >= next_sample:
                stats = previous.snapshot()
                stats.update(running=len(running), pending=len(pending), slots=slots)
                recent.append(stats)
                recent = recent[-15:]
                path = STUDY / 'resource_samples.csv'
                exists = path.exists()
                with path.open('a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(stats))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(stats)
                if time.monotonic() - start > 30 and len(recent) >= 10 and pending and len(running) >= slots:
                    if (np.mean([x['gpu_util'] for x in recent]) < 80 and np.mean([x['cpu_percent'] for x in recent]) < 75
                            and stats['gpu_free_mib'] > cfg['reserve_gpu_mib'] + cfg['new_job_gpu_mib']):
                        slots = min(slots + 1, cfg['max_parallel'])
                next_sample = time.monotonic() + 2
            while pending and len(running) < slots:
                if previous.cohort.resources()['gpu_free_mib'] < cfg['reserve_gpu_mib'] + cfg['new_job_gpu_mib'] or (running and stats['cpu_percent'] > 90):
                    break
                job = pending.pop(0)
                f = (logs / (job['id'] + '.log')).open('a', buffering=1)
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                p = subprocess.Popen([sys.executable, __file__, 'worker', '--job', job['id']], cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
                running[p.pid] = (p, job, f)
                print('[launch]', job['id'], p.pid, flush=True)
            if completed:
                commit('results: complete new-seed AC LS jobs ' + ','.join(completed))
            core.save_json(STUDY / 'status.json', dict(phase='training', total=15, completed=15 - len(pending) - len(running),
                pending=len(pending), slots=slots, resources=stats, running=[dict(job=j['id'], pid=p.pid) for p, j, _ in running.values()], updated_at=core.timestamp()))
            if pending or running:
                time.sleep(.5)
        core.save_json(STUDY / 'results.json', [core.read_json(STUDY / 'runs' / j['id'] / 'validation_complete.json') for j in jobs])
        commit('results: complete AC LS replication for independent audit')
        core.save_json(STUDY / 'status.json', dict(phase='complete_needs_independent_audit', total=15, completed=15, updated_at=core.timestamp()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'check', 'worker', 'run'])
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'worker':
        worker(args.job)
    else:
        run()

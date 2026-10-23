#!/usr/bin/env python3
"""Frozen patient-group OOF grading and three-arm CLE data study.

Run prepare, then run. The supervisor is resumable and starts only its own jobs.
Large images/caches/checkpoints are ignored by Git; manifests and results are not.
"""
import argparse
import collections
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'experiments/data_study'
CACHE = ROOT / '.cache_data_study'
WEIGHTS = ROOT / 'weights/data_study'
GIT = '/home/login/miniconda3/bin/git'
PRETRAINED = Path('/home/login/data/cle/sunfuxiang/classifyML/pth/resnet34-333f7ec4.pth')
META_ROOT = Path('/home/login/data/cle/sunfuxiang/csv')
CLASSES = ['Benign_lesions', 'Malignant_lesions', 'body_of_stomach_normal']
sys.path.insert(0, str(ROOT))


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit(message):
    subprocess.run([GIT, '-C', str(ROOT), 'add', 'experiments', 'STUDY_PROTOCOL.md', '.gitignore'], check=True)
    changed = subprocess.run([GIT, '-C', str(ROOT), 'diff', '--cached', '--quiet']).returncode
    if changed:
        subprocess.run([GIT, '-C', str(ROOT), 'commit', '-m', message], check=True)
    return subprocess.check_output([GIT, '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def validate_split(records, train, inner, outer):
    sets = [set(x) for x in (train, inner, outer)]
    groups = [{records[i]['group_id'] for i in x} for x in sets]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not sets[a] & sets[b], 'sample overlap'
        assert not groups[a] & groups[b], 'patient overlap'
    for x in sets:
        assert {records[i]['label'] for i in x} == {0, 1, 2}, 'missing class'


def prepare():
    import numpy as np
    from PIL import Image
    from sklearn.model_selection import StratifiedGroupKFold
    import torch
    import torchvision
    import sklearn
    STUDY.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if (STUDY / 'config.json').exists():
        validate_prepared()
        print('Existing frozen protocol validated; not regenerating folds.', flush=True)
        return
    metadata = {}
    for split in ('train', 'val', 'test'):
        with open(META_ROOT / f'{split}_processed.csv', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                sid = row['sample_id']
                assert sid not in metadata, 'nonunique upstream sample ID'
                metadata[sid] = (row['leakage_group_id'], row['split'])
    records = []
    hashes = {}
    for split in ('train', 'val', 'test'):
        for label, cls in enumerate(CLASSES):
            for p in sorted((ROOT / 'datasets/cle' / split / cls).iterdir()):
                if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                    continue
                sid = re.search(r'S_[0-9a-f]{12}', p.name).group()
                group, original_split = metadata[sid]
                assert original_split == split, (sid, original_split, split)
                sha = digest(p)
                assert sha not in hashes, 'exact duplicate image: ' + str(p)
                hashes[sha] = p
                records.append(dict(index=len(records), sample_id=sid, split=split,
                                    label=label, class_name=cls, group_id=group,
                                    path=str(p.relative_to(ROOT)), sha256=sha))
    assert len({r['sample_id'] for r in records}) == len(records)
    groups = {s: {r['group_id'] for r in records if r['split'] == s} for s in ('train', 'val', 'test')}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not groups[a] & groups[b], f'{a}/{b} patient leakage'
    save_json(STUDY / 'manifest.json', records)
    train_ids = np.array([r['index'] for r in records if r['split'] == 'train'])
    val_ids = [r['index'] for r in records if r['split'] == 'val']
    test_ids = [r['index'] for r in records if r['split'] == 'test']
    jobs = []
    for seed in (42, 43, 44):
        y = np.array([records[i]['label'] for i in train_ids])
        g = np.array([records[i]['group_id'] for i in train_ids])
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        coverage = collections.Counter()
        for fold, (rest_pos, held_pos) in enumerate(outer.split(train_ids, y, g)):
            rest = train_ids[rest_pos]
            held = train_ids[held_pos]
            # The outer patients never participate in checkpoint selection.
            inner_cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed * 100 + fold)
            inner_y = [records[i]['label'] for i in rest]
            inner_g = [records[i]['group_id'] for i in rest]
            selected = None
            for fit_pos, sel_pos in inner_cv.split(rest, inner_y, inner_g):
                if all({records[i]['label'] for i in z} == {0, 1, 2}
                       for z in (rest[fit_pos], rest[sel_pos], held)):
                    selected = (rest[fit_pos], rest[sel_pos])
                    break
            assert selected is not None, 'cannot create three-class inner selection split'
            fit, select = selected
            validate_split(records, fit, select, held)
            coverage.update(int(i) for i in held)
            jobs.append(dict(id=f'oof_s{seed}_f{fold}', phase='oof', seed=seed, fold=fold,
                             train=[int(i) for i in fit], select=[int(i) for i in select],
                             score=[int(i) for i in held], epochs=60, arm='oof'))
        assert set(coverage) == set(train_ids) and set(coverage.values()) == {1}
    config = dict(version=1, created_at=timestamp(), model='resnet34', seeds=[42, 43, 44],
                  folds=5, epochs=60, batch_size=32, img_size=224, resize=256,
                  optimizer='AdamW', lr=1e-4, weight_decay=0.05, scheduler='cosine_to_zero',
                  amp=True, workers=2, max_parallel=2, early_stopping=False,
                  selection='inner_macro_f1_for_oof; fixed_val_macro_f1_for_final',
                  grade_rules=dict(A=[0.95, 0.95], B=[0.85, 0.85], D=[0.60, 0.60], E=[0.30, 0.30]),
                  confidence='mean OOF probability of existing label (not predicted-class confidence)',
                  grade_weights=dict(A=1.0, B=1.0, C=0.75, D=0.25, E=0.10),
                  weight_normalization='within each original class to mean 1; loss is mean(w * CE)',
                  arms=['full', 'ac_only', 'grade_weighted'],
                  test_policy='complete all 9 final trainings before frozen test evaluation; no test filtering',
                  val_indices=val_ids, test_indices=test_ids,
                  pretrained=str(PRETRAINED), pretrained_sha256=digest(PRETRAINED),
                  manifest_sha256=digest(STUDY / 'manifest.json'),
                  versions=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                sklearn=sklearn.__version__, numpy=np.__version__))
    save_json(STUDY / 'config.json', config)
    save_json(STUDY / 'oof_jobs.json', jobs)
    # Cache only a label-independent Resize(256,256), equivalent to original PIL preprocessing.
    target = CACHE / 'resized_uint8.npy'
    tmp = CACHE / 'resized_uint8.tmp.npy'
    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(records), 256, 256, 3))
    for i, r in enumerate(records):
        with Image.open(ROOT / r['path']) as im:
            arr[i] = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
        if i % 500 == 0:
            print(f'[cache] {i}/{len(records)}', flush=True)
    arr.flush()
    del arr
    os.replace(tmp, target)
    save_json(CACHE / 'meta.json', dict(manifest_sha256=config['manifest_sha256'], count=len(records),
                                       transform='PIL RGB Resize(256,256) BILINEAR', cache_sha256=digest(target)))
    validate_prepared()
    summary = {s: dict(images=sum(r['split'] == s for r in records), groups=len(groups[s]),
                       classes={c: sum(r['split'] == s and r['class_name'] == c for r in records) for c in CLASSES})
               for s in ('train', 'val', 'test')}
    save_json(STUDY / 'dataset_summary.json', summary)
    print(json.dumps(summary), flush=True)
    git_commit('data: freeze image hashes, patient-isolated OOF folds and study protocol')


def validate_prepared():
    import numpy as np
    records = read_json(STUDY / 'manifest.json')
    config = read_json(STUDY / 'config.json')
    assert digest(STUDY / 'manifest.json') == config['manifest_sha256']
    assert digest(PRETRAINED) == config['pretrained_sha256']
    for job in read_json(STUDY / 'oof_jobs.json'):
        validate_split(records, job['train'], job['select'], job['score'])
    cache = read_json(CACHE / 'meta.json')
    assert cache['manifest_sha256'] == config['manifest_sha256']
    assert digest(CACHE / 'resized_uint8.npy') == cache['cache_sha256']
    arr = np.load(CACHE / 'resized_uint8.npy', mmap_mode='r')
    assert arr.shape == (len(records), 256, 256, 3) and arr.dtype == np.uint8


def build_dataset(indices, training=False, sample_weights=None):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    records = read_json(STUDY / 'manifest.json')
    transform = T.Compose(([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
                            T.RandomCrop(224)] if training else [T.CenterCrop(224)]) +
                          [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.array = np.load(CACHE / 'resized_uint8.npy', mmap_mode='r')

        def __len__(self):
            return len(indices)

        def __getitem__(self, position):
            idx = indices[position]
            image = Image.fromarray(self.array[idx])
            return transform(image), records[idx]['label'], float(sample_weights.get(idx, 1.0) if sample_weights else 1.0), idx

    return ImageDataset()


def loader(indices, training=False, sample_weights=None, seed=42):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(build_dataset(indices, training, sample_weights), batch_size=32,
                                       shuffle=training, num_workers=2, pin_memory=True,
                                       persistent_workers=True, generator=generator)


def new_model(seed):
    import torch
    from cle_fair_common import build_model
    set_seed(seed)
    model = build_model('resnet34', 3)
    # The official 2017 ResNet weights use PyTorch's legacy tar format. Only
    # this exact published artifact may use the legacy deserializer.
    assert digest(PRETRAINED) == '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'
    state = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
    for key in ['fc.weight', 'fc.bias']:
        state.pop(key)
    inc = model.load_state_dict(state, strict=False)
    assert set(inc.missing_keys) == {'fc.weight', 'fc.bias'} and not inc.unexpected_keys, inc
    return model.cuda()


def infer(model, data_loader):
    import numpy as np
    import torch
    model.eval()
    probs, labels, indices = [], [], []
    with torch.inference_mode():
        for x, y, _, idx in data_loader:
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x.cuda(non_blocking=True))
            probs.append(logits.float().softmax(1).cpu().numpy())
            labels.extend(y.tolist())
            indices.extend(idx.tolist())
    return np.concatenate(probs), np.array(labels), indices


def prediction_rows(probs, labels, indices):
    records = read_json(STUDY / 'manifest.json')
    return [dict(index=i, sample_id=records[i]['sample_id'], group_id=records[i]['group_id'],
                 label=int(y), pred=int(p.argmax()), p0=float(p[0]), p1=float(p[1]), p2=float(p[2]))
            for p, y, i in zip(probs, labels, indices)]


def metrics(probs, labels):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
    pred = probs.argmax(1)
    return dict(macro_f1=float(f1_score(labels, pred, average='macro', labels=[0, 1, 2], zero_division=0)),
                accuracy=float(accuracy_score(labels, pred)),
                class_f1=f1_score(labels, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
                confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist(),
                auroc=float(roc_auc_score(labels, probs, multi_class='ovr', average='macro')))


def signature(job):
    return hashlib.sha256((json.dumps(job, sort_keys=True) + digest(STUDY / 'config.json') +
                           digest(__file__)).encode()).hexdigest()


def run_worker(job_id):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    jobs = read_json(STUDY / ('oof_jobs.json' if job_id.startswith('oof_') else 'final_jobs.json'))
    job = next(j for j in jobs if j['id'] == job_id)
    config = read_json(STUDY / 'config.json')
    assert digest(PRETRAINED) == config['pretrained_sha256']
    seed = job['seed']
    model = new_model(seed)
    run_dir = STUDY / 'runs' / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckdir = WEIGHTS / job_id
    ckdir.mkdir(parents=True, exist_ok=True)
    ckpt = ckdir / 'model_best.pth'
    weights = {int(k): v for k, v in job.get('sample_weights', {}).items()}
    train_loader = loader(job['train'], True, weights, seed)
    val_loader = loader(job['select'], False, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=job['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    best, best_epoch = -1.0, -1
    history = []
    started = time.time()
    meta = dict(job_id=job_id, signature=signature(job), seed=seed, arm=job['arm'],
                train_count=len(job['train']), select_count=len(job['select']),
                created_at=timestamp(), git_commit=subprocess.check_output([GIT, '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                pretrained_sha256=config['pretrained_sha256'])
    save_json(run_dir / 'run_meta.json', meta)
    for epoch in range(job['epochs']):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y, w, _ in train_loader:
            x, y, w = x.cuda(non_blocking=True), y.cuda(non_blocking=True), w.cuda(non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)
        p, y, _ = infer(model, val_loader)
        val_f1 = float(f1_score(y, p.argmax(1), average='macro', labels=[0, 1, 2], zero_division=0))
        scheduler.step()
        if val_f1 > best:
            best, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), ckpt.with_suffix('.tmp'))
            os.replace(ckpt.with_suffix('.tmp'), ckpt)
        history.append(dict(epoch=epoch, train_loss=total_loss/n, train_acc=correct/n,
                            val_macro_f1=val_f1, lr=optimizer.param_groups[0]['lr'],
                            elapsed_seconds=round(time.time()-started, 2)))
        save_csv(run_dir / 'train_log.csv', history)
        print(f'[{job_id}] epoch={epoch+1}/60 train_acc={correct/n:.4f} val_f1={val_f1:.4f} best={best:.4f} elapsed={time.time()-started:.1f}s', flush=True)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
    if job['phase'] == 'oof':
        p, y, idx = infer(model, loader(job['score'], False, seed=seed))
        save_csv(run_dir / 'oof_predictions.csv', prediction_rows(p, y, idx))
        meta['outer_metrics'] = metrics(p, y)
    meta.update(completed_at=timestamp(), seconds=round(time.time()-started, 2),
                best_select_f1=best, best_epoch=best_epoch,
                checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=digest(ckpt),
                epochs_completed=len(history))
    save_json(run_dir / 'complete.json', meta)


def grade(correct_rate, confidence):
    if correct_rate >= .95 and confidence >= .95:
        return 'A'
    if correct_rate >= .85 and confidence >= .85:
        return 'B'
    if correct_rate < .30 or confidence < .30:
        return 'E'
    if correct_rate < .60 or confidence < .60:
        return 'D'
    return 'C'


def aggregate_grades():
    import numpy as np
    records = read_json(STUDY / 'manifest.json')
    config = read_json(STUDY / 'config.json')
    pred = collections.defaultdict(list)
    for job in read_json(STUDY / 'oof_jobs.json'):
        with open(STUDY / 'runs' / job['id'] / 'oof_predictions.csv') as f:
            for r in csv.DictReader(f):
                idx = int(r['index'])
                assert idx in job['score']
                assert int(r['label']) == records[idx]['label']
                pred[idx].append((job['seed'], int(r['pred']), [float(r[f'p{i}']) for i in range(3)]))
    rows = []
    for r in records:
        if r['split'] != 'train':
            continue
        parts = sorted(pred[r['index']])
        assert [p[0] for p in parts] == [42, 43, 44], 'missing or repeated OOF predictions'
        probs = np.array([p[2] for p in parts])
        correct = np.array([p[1] == r['label'] for p in parts])
        conf = probs[:, r['label']]
        gr = grade(correct.mean(), conf.mean())
        rows.append(dict(index=r['index'], sample_id=r['sample_id'], group_id=r['group_id'],
                         class_name=r['class_name'], label=r['label'], path=r['path'],
                         grade=gr, pred_correct_rate=float(correct.mean()), mean_confidence=float(conf.mean()),
                         std_confidence=float(conf.std()), mean_entropy=float((-probs*np.log(probs+1e-12)).sum(1).mean()),
                         p0=float(probs[:,0].mean()), p1=float(probs[:,1].mean()), p2=float(probs[:,2].mean()),
                         num_predictions=3, raw_weight=config['grade_weights'][gr]))
    save_csv(STUDY / 'train_grades.csv', rows)
    distribution = {c: dict(collections.Counter(r['grade'] for r in rows if r['class_name'] == c)) for c in CLASSES}
    save_json(STUDY / 'grade_summary.json', dict(total=len(rows), by_class=distribution,
              counts=dict(collections.Counter(r['grade'] for r in rows)),
              retained_ac=sum(r['grade'] in 'ABC' for r in rows),
              note='OOF agreement/learnability grades; not verified clinical label correctness.'))
    final = []
    for arm in config['arms']:
        retained = [r for r in rows if arm != 'ac_only' or r['grade'] in 'ABC']
        assert {r['label'] for r in retained} == {0,1,2}
        sample_weights = {}
        if arm == 'grade_weighted':
            means = {c: np.mean([r['raw_weight'] for r in rows if r['label'] == c]) for c in range(3)}
            sample_weights = {str(r['index']): float(r['raw_weight']/means[r['label']]) for r in rows}
            for c in range(3):
                assert abs(np.mean([sample_weights[str(r['index'])] for r in rows if r['label']==c])-1)<1e-10
        for seed in config['seeds']:
            final.append(dict(id=f'{arm}_s{seed}', phase='final', arm=arm, seed=seed,
                              train=[r['index'] for r in retained], select=config['val_indices'],
                              score=config['test_indices'], epochs=60, sample_weights=sample_weights))
    save_json(STUDY / 'final_jobs.json', final)
    git_commit('data: complete patient-level three-seed A-E grading and freeze three-arm training manifests')
    return final


def complete(job):
    p = STUDY / 'runs' / job['id'] / 'complete.json'
    if not p.exists():
        return False
    result = read_json(p)
    assert result['signature'] == signature(job), 'completed job configuration/code changed; refuse stale artifact reuse'
    assert digest(ROOT / result['checkpoint']) == result['checkpoint_sha256'], 'checkpoint hash changed'
    return True


def status(phase, **extra):
    save_json(STUDY / 'status.json', dict(updated_at=timestamp(), phase=phase, **extra))


def run_jobs(jobs, phase, maximum):
    pending = [j for j in jobs if not complete(j)]
    running = {}
    retries = collections.Counter()
    logs = STUDY / 'logs'
    logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running) < maximum:
            job = pending.pop(0)
            log = open(logs / (job['id'] + '.log'), 'a', buffering=1)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), 'worker', '--job', job['id']],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[proc.pid] = (proc, job, log)
            print(f'[launch] {job["id"]} pid={proc.pid}', flush=True)
        status(phase, total=len(jobs), completed=len(jobs)-len(pending)-len(running),
               pending=len(pending), running=[dict(job=j['id'], pid=p.pid) for p,j,_ in running.values()])
        for pid, (proc, job, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[pid]
            if rc or not complete(job):
                retries[job['id']] += 1
                if retries[job['id']] <= 1:
                    print(f'[retry] {job["id"]} rc={rc}', flush=True)
                    pending.append(job)
                else:
                    # Do not terminate unrelated processes; wait for own active jobs before failing.
                    for p, _, lf in running.values():
                        p.wait()
                        lf.close()
                    raise RuntimeError(f'{job["id"]} failed twice; inspect {logs/job["id"]}')
            else:
                print(f'[done] {job["id"]}', flush=True)
                git_commit(f'results: finish {job["id"]} with audited checkpoint and metrics')
        if pending or running:
            time.sleep(15)


def final_evaluation(jobs):
    import numpy as np
    import torch
    results = []
    for job in jobs:
        run_dir = STUDY / 'runs' / job['id']
        completion = read_json(run_dir / 'complete.json')
        result_file = run_dir / 'test_result.json'
        if result_file.exists():
            result = read_json(result_file)
            assert result['checkpoint_sha256'] == completion['checkpoint_sha256']
        else:
            model = new_model(job['seed'])
            model.load_state_dict(torch.load(ROOT/completion['checkpoint'], map_location='cpu', weights_only=True), strict=True)
            p, y, idx = infer(model, loader(job['score'], False, seed=job['seed']))
            save_csv(run_dir / 'test_predictions.csv', prediction_rows(p,y,idx))
            result = dict(job_id=job['id'], arm=job['arm'], seed=job['seed'], n=len(y),
                          checkpoint_sha256=completion['checkpoint_sha256'], evaluated_at=timestamp(), **metrics(p,y))
            save_json(result_file, result)
            del model
            torch.cuda.empty_cache()
        results.append(result)
        git_commit('results: record frozen complete-test evaluation for '+job['id'])
    save_json(STUDY/'test_results.json',results)
    rows = [dict(arm=r['arm'],seed=r['seed'],macro_f1=r['macro_f1'],accuracy=r['accuracy'],
                 benign_f1=r['class_f1'][0],malignant_f1=r['class_f1'][1],normal_f1=r['class_f1'][2],
                 checkpoint_sha256=r['checkpoint_sha256']) for r in results]
    save_csv(STUDY/'test_results.csv',rows)
    # Fixed fitted models: patient-cluster bootstrap of the paired average over seeds.
    loaded = {}
    for job in jobs:
        with open(STUDY/'runs'/job['id']/'test_predictions.csv') as f:
            loaded[job['id']] = sorted(list(csv.DictReader(f)),key=lambda r:int(r['index']))
    first = next(iter(loaded.values()))
    ids=[r['index'] for r in first]
    groups=sorted({r['group_id'] for r in first})
    gi=np.array([groups.index(r['group_id']) for r in first])
    rng=np.random.default_rng(20261024)
    bw=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
    def group_cm(rows):
        assert [r['index'] for r in rows] == ids
        cm=np.zeros((len(groups),3,3))
        np.add.at(cm,(gi,[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1)
        return cm
    def f1(cm):
        denom=cm.sum(-1)+cm.sum(-2)
        return np.mean(np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),denom,
                                  out=np.zeros_like(denom),where=denom!=0),axis=-1)
    comparisons={}
    for arm in ('ac_only','grade_weighted'):
        ds=[]
        observed=[]
        for seed in (42,43,44):
            a=group_cm(loaded[f'full_s{seed}']);b=group_cm(loaded[f'{arm}_s{seed}'])
            ds.append(f1(np.einsum('bg,gij->bij',bw,b))-f1(np.einsum('bg,gij->bij',bw,a)))
            observed.append(float(f1(b.sum(0))-f1(a.sum(0))))
        comparisons[arm]=dict(seed_deltas=observed,mean_delta=float(np.mean(observed)),
                              patient_cluster95=np.percentile(np.mean(ds,axis=0),[2.5,97.5]).tolist(),
                              caveat='Conditional on these three trained pairs; excludes grading and training-data resampling uncertainty.')
    save_json(STUDY/'paired_comparisons.json',comparisons)
    lines=['# Frozen CLE data study results','',f'Completed: {timestamp()}','',
           'All methods use the same ResNet34, three seeds, 60 epochs, preprocessing and complete test set.',
           'A-E grades measure OOF agreement/learnability, not verified label correctness.','',
           '| Arm | Mean macro-F1 | Seed SD | Mean malignant F1 |','|---|---:|---:|---:|']
    for arm in ('full','ac_only','grade_weighted'):
        rs=[r for r in results if r['arm']==arm]
        lines.append(f'| {arm} | {np.mean([r["macro_f1"] for r in rs]):.6f} | {np.std([r["macro_f1"] for r in rs],ddof=1):.6f} | {np.mean([r["class_f1"][1] for r in rs]):.6f} |')
    lines+=['','Paired comparisons (fraction units, multiply by 100 for percentage points):','',
            '```json',json.dumps(comparisons,indent=2),'```','',
            'The existing test set has previously been used in historical model selection. These are controlled comparisons, not a new untouched external validation.',
            'No labels were changed; no validation/test images were removed; no method or threshold was selected using these test results.']
    (STUDY/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    git_commit('results: complete three-arm data study with patient-paired uncertainty report')


def run(maximum):
    STUDY.mkdir(parents=True,exist_ok=True)
    with open(STUDY/'pipeline.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            validate_prepared()
            run_jobs(read_json(STUDY/'oof_jobs.json'),'oof_grading',maximum)
            final=read_json(STUDY/'final_jobs.json') if (STUDY/'final_jobs.json').exists() else aggregate_grades()
            for arm in ('full','ac_only','grade_weighted'):
                run_jobs([j for j in final if j['arm']==arm],arm+'_training',maximum)
            status('frozen_test_evaluation',total=9)
            final_evaluation(final)
            head=subprocess.check_output([GIT,'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
            status('complete',completed=24,total=24,git_commit=head,report='experiments/data_study/FINAL_REPORT.md')
        except Exception as e:
            status('failed',error=repr(e))
            raise


def self_check():
    import numpy as np
    import torch
    import torch.nn.functional as F
    assert [grade(1,.98),grade(1,.90),grade(2/3,.7),grade(1/3,.7),grade(0,.99)]==list('ABCDE')
    logits=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
    y=torch.tensor([0,2])
    assert torch.allclose((F.cross_entropy(logits,y,reduction='none')*torch.ones(2)).mean(),F.cross_entropy(logits,y))
    fake=[dict(group_id='p1',label=i) for i in range(3)]*3
    try:
        validate_split(fake,[0,1,2],[3,4,5],[6,7,8])
    except AssertionError:
        pass
    else:
        raise AssertionError('group leakage was not rejected')
    validate_prepared()
    set_seed(42)
    a=new_model(42).cpu().state_dict()
    b=new_model(42).cpu().state_dict()
    assert all(torch.equal(a[k],b[k]) for k in a), 'paired seed initialization mismatch'
    records=read_json(STUDY/'manifest.json')
    ids=[r['index'] for r in records if r['split']=='train'][:32]
    model=new_model(42)
    batch=next(iter(loader(ids,True,seed=42)))
    with torch.autocast('cuda',dtype=torch.float16):
        logits=model(batch[0].cuda())
        loss=F.cross_entropy(logits,batch[1].cuda())
    loss.backward()
    assert torch.isfinite(loss) and logits.shape==(32,3)
    print('PASS: grade boundaries, neutral weighting, patient leak guard, manifest/cache hashes, paired initialization, AMP forward/backward.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['prepare','run','worker','check'])
    ap.add_argument('--job')
    ap.add_argument('--max-parallel',type=int,default=2)
    args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='worker':run_worker(args.job)
    elif args.command=='check':self_check()
    else:run(args.max_parallel)

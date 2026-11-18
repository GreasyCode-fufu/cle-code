#!/usr/bin/env python3
"""R15: new-split replication of the unconfirmed nested AC selection signal."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
import run_architecture_study as arch
import frozen_feature_grader as grader
core = arch.core
DATA = ROOT/'experiments/data_study'
SOURCE = ROOT/'experiments/fusion_confirmation'
STUDY = ROOT/'experiments/grade_confirmation'
FEATURES = ROOT/'.cache_representation_study/features.npz'
FEATURE_META = FEATURES.with_suffix('.json')
ARMS = ['ac_only', 'random_ac']
SEEDS = [45, 46, 47]
COMPARISONS = [('ac_only', 'full_ce'), ('ac_only', 'random_ac')]
arch.STUDY = STUDY
arch.__file__ = __file__
core.STUDY = STUDY
core.WEIGHTS = ROOT/'weights/grade_confirmation'
core.new_model = arch.ORIGINAL_MODEL
SAVE = core.save_json
STAGE = 'nested_grader_fitting'


def save_json(path, value):
    if Path(path) == STUDY/'status.json' and value.get('phase') == 'round1_training':
        value = dict(value, phase='round15_'+STAGE, round=15)
    SAVE(path, value)


core.save_json = save_json


def sources():
    import sklearn
    return dict(driver=core.digest(__file__), helper=core.digest(grader.__file__),
                protocol=core.digest(ROOT/'experiments/ROUND15_PROTOCOL.md'),
                prior_nested_driver=core.digest(ROOT/'experiments/run_nested_grade_study.py'),
                trainer=core.digest(core.__file__), supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
                backbone=core.digest(ROOT/'cle_fair_common.py'), sklearn=sklearn.__version__, numpy=np.__version__)


def original_signature(job):
    return hashlib.sha256((json.dumps(job, sort_keys=True)+core.digest(SOURCE/'config.json')+
                           core.digest(core.__file__)).encode()).hexdigest()


def grade_signature(job):
    return hashlib.sha256((json.dumps(job, sort_keys=True)+core.digest(STUDY/'grading_config.json')).encode()).hexdigest()


def verify_grading():
    cfg = core.read_json(STUDY/'grading_config.json')
    assert cfg['sources'] == sources()
    for path, key in [(STUDY/'manifest.json', 'manifest_sha256'), (STUDY/'grading_jobs.json', 'grading_jobs_sha256'),
                      (SOURCE/'config.json', 'source_config_sha256'), (SOURCE/'final_jobs.json', 'source_jobs_sha256'),
                      (SOURCE/'folds.json', 'source_folds_sha256'),
                      (ROOT/'experiments/nested_grade_study/completion_audit.json', 'prior_nested_audit_sha256'),
                      (SOURCE/'completion_audit.json', 'source_audit_sha256'),
                      (FEATURE_META, 'feature_meta_sha256'), (core.CACHE/'meta.json', 'cache_meta_sha256'),
                      (ROOT/'experiments/representation_study/completion_audit.json', 'r9_audit_sha256')]:
        assert core.digest(path) == cfg[key], str(path)
    for name, sha in cfg['baseline_file_hashes'].items():
        assert core.digest(STUDY/name) == sha
    return cfg


def verify():
    verify_grading()
    cfg = core.read_json(STUDY/'config.json')
    assert cfg['grading_config_sha256'] == core.digest(STUDY/'grading_config.json')
    assert cfg['jobs_sha256'] == core.digest(STUDY/'final_jobs.json')
    for name, sha in cfg['grade_file_hashes'].items():
        assert core.digest(STUDY/name) == sha
    return cfg


arch.verify = verify


def load_features():
    cfg = verify_grading()
    assert core.digest(FEATURES) == cfg['feature_sha256']
    with np.load(FEATURES, allow_pickle=False) as data:
        assert np.array_equal(data['indices'], np.arange(2386))
        assert data['features'].shape == (2386, 1280)
        x = data['features'][:, 512:1280].astype(np.float64)
    assert np.isfinite(x).all()
    return x


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY/'grading_config.json').exists():
        verify_grading()
        return
    original = [j for j in core.read_json(SOURCE/'final_jobs.json') if j['arm']=='resnet_bn_train']
    assert len(original) == 15
    shutil.copyfile(DATA/'manifest.json', STUDY/'manifest.json')
    hashes = {}
    index = []
    jobs = []
    for old in original:
        comp = core.read_json(SOURCE/'runs'/old['id']/'complete.json')
        assert comp['signature'] == original_signature(old) and comp['epochs_completed'] == 60
        assert core.digest(ROOT/comp['checkpoint']) == comp['checkpoint_sha256']
        out = STUDY/'baselines'/old['id']
        out.mkdir(parents=True, exist_ok=True)
        for name in ['complete.json', 'run_meta.json', 'train_log.csv', 'oof_predictions.csv']:
            shutil.copyfile(SOURCE/'runs'/old['id']/name, out/name)
            hashes[str((out/name).relative_to(STUDY))] = core.digest(out/name)
        index.append(dict(source_job=old, source_signature=comp['signature'], checkpoint_sha256=comp['checkpoint_sha256']))
        jobs.append(dict(old, id=f'grade_s{old["seed"]}_f{old["fold"]}', source_job_id=old['id']))
    core.save_json(STUDY/'baseline_index.json', index)
    hashes['baseline_index.json'] = core.digest(STUDY/'baseline_index.json')
    core.save_json(STUDY/'grading_jobs.json', jobs)
    meta = core.read_json(FEATURE_META)
    assert core.digest(FEATURES) == meta['sha256'] and meta['shape'] == [2386, 1280]
    cfg = dict(round=15, created_at=core.timestamp(), sources=sources(), seeds=SEEDS, grader_replicates=grader.GRADER_SEEDS,
               manifest_sha256=core.digest(STUDY/'manifest.json'), grading_jobs_sha256=core.digest(STUDY/'grading_jobs.json'),
               source_config_sha256=core.digest(SOURCE/'config.json'), source_jobs_sha256=core.digest(SOURCE/'final_jobs.json'),
               source_folds_sha256=core.digest(SOURCE/'folds.json'),
               prior_nested_audit_sha256=core.digest(ROOT/'experiments/nested_grade_study/completion_audit.json'),
               source_audit_sha256=core.digest(SOURCE/'completion_audit.json'),
               feature_meta_sha256=core.digest(FEATURE_META), feature_sha256=core.digest(FEATURES), feature_columns=[512,1280],
               cache_meta_sha256=core.digest(core.CACHE/'meta.json'), baseline_file_hashes=hashes,
               r9_audit_sha256=core.digest(ROOT/'experiments/representation_study/completion_audit.json'),
               pretrained_sha256=core.digest(core.PRETRAINED), active_indices=list(range(2386)),
               linear_grader_fits=225, new_neural_trainings_max=30, reused_baselines=15,
               policy='Only fit patients graded; inner/outer excluded. No original fixed val/test, label changes or outer filtering.')
    core.save_json(STUDY/'grading_config.json', cfg)
    core.save_json(STUDY/'status.json', dict(phase='round15_prepared', round=15, grading_units=15, updated_at=core.timestamp()))
    arch.commit('research: freeze strict nested grading protocol and 15 signed45-47 source folds')


def grade_done(job):
    path = STUDY/'grades'/job['id']/'complete.json'
    if not path.exists():
        return False
    value = core.read_json(path)
    assert value['signature'] == grade_signature(job)
    for name, sha in value['file_hashes'].items():
        assert core.digest(path.parent/name) == sha
    for name, sha in value['head_hashes'].items():
        assert core.digest(ROOT/name) == sha
    return True


def grade_worker(job):
    if grade_done(job):
        return
    features = load_features()
    records = core.read_json(STUDY/'manifest.json')
    labels = np.array([r['label'] for r in records])
    groups = np.array([r['group_id'] for r in records])
    out = STUDY/'grades'/job['id']
    out.mkdir(parents=True, exist_ok=True)
    head_dir = core.WEIGHTS/'graders'/job['id']
    head_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'head_manifest.json'
    manifest = core.read_json(manifest_path) if manifest_path.exists() else dict(signature=grade_signature(job), heads={})
    assert manifest['signature'] == grade_signature(job)
    predicted = collections.defaultdict(list)
    all_rows = []
    for replicate, fold, fit, held in grader.partitions(job, labels, groups):
        key = f'r{replicate}_f{fold}'
        path = head_dir/(key+'.npz')
        if key in manifest['heads']:
            assert core.digest(path) == manifest['heads'][key]['sha256']
            with np.load(path, allow_pickle=False) as h:
                assert np.array_equal(h['fit_indices'], fit) and np.array_equal(h['held_indices'], held)
                p = grader.softmax(((features[held]-h['mean'])/h['scale'])@h['coef'].T+h['intercept'])
        else:
            p = grader.fit_head(features, labels, fit, held, path)
            manifest['heads'][key] = dict(path=str(path.relative_to(ROOT)), sha256=core.digest(path),
                                        fit_indices=fit.tolist(), held_indices=held.tolist(), replicate=replicate, fold=fold)
            core.save_json(manifest_path, manifest)
        for i, probability in zip(held, p):
            predicted[int(i)].append((replicate, probability))
            all_rows.append(dict(index=int(i), replicate=replicate, nested_fold=fold, label=int(labels[i]),
                                 pred=int(probability.argmax()), **{f'p{k}': float(probability[k]) for k in range(3)}))
    assert set(predicted) == set(job['train']) and len(manifest['heads']) == 15
    grade_rows = []
    grades = {}
    for i in job['train']:
        parts = sorted(predicted[i], key=lambda x: x[0])
        assert [r[0] for r in parts] == grader.GRADER_SEEDS
        p = np.array([r[1] for r in parts])
        correctness = float(np.mean(p.argmax(1) == labels[i]))
        confidence = float(p[:, labels[i]].mean())
        grades[i] = core.grade(correctness, confidence)
        grade_rows.append(dict(index=i, sample_id=records[i]['sample_id'], group_id=groups[i], label=int(labels[i]),
                               correct_rate=correctness, confidence=confidence, grade=grades[i],
                               **{f'p{k}': float(p[:,k].mean()) for k in range(3)}))
    student_plans = grader.plans(job, records, grades)
    core.save_csv(out/'grades.csv', grade_rows)
    core.save_csv(out/'crossfit_predictions.csv', all_rows)
    core.save_json(out/'student_plans.json', student_plans)
    counts = {str(c): dict(collections.Counter(grades[i] for i in job['train'] if labels[i] == c)) for c in [0,1,2]}
    core.save_json(out/'grade_summary.json', dict(n=len(grades), per_class=counts,
                   ac_per_class={str(c): int(sum(labels[i] == c for i in student_plans['ac_only']['train'])) for c in [0,1,2]}))
    names = ['head_manifest.json','grades.csv','crossfit_predictions.csv','student_plans.json','grade_summary.json']
    core.save_json(out/'complete.json', dict(signature=grade_signature(job), linear_fits=15, graded_n=len(grades),
                   file_hashes={name: core.digest(out/name) for name in names},
                   head_hashes={h['path']: h['sha256'] for h in manifest['heads'].values()}, completed_at=core.timestamp()))


def effective_key(job):
    weights = job.get('sample_weights', {})
    return json.dumps([job['seed'], job['train'], job['select'], job['score'],
                       [weights.get(str(i), 1.) for i in job['train']]], sort_keys=True)


def freeze_training():
    if (STUDY/'config.json').exists():
        verify()
        return
    cfg = verify_grading()
    jobs = []
    hashes = {}
    for unit in core.read_json(STUDY/'grading_jobs.json'):
        assert grade_done(unit)
        out = STUDY/'grades'/unit['id']
        for path in out.iterdir():
            assert path.is_file()
            hashes[str(path.relative_to(STUDY))] = core.digest(path)
        old = next(v['source_job'] for v in core.read_json(STUDY/'baseline_index.json') if v['source_job']['id'] == unit['source_job_id'])
        seen = {effective_key(old): dict(kind='baseline', id=old['id'])}
        for arm, plan in core.read_json(out/'student_plans.json').items():
            if arm not in ARMS: continue
            j = dict(old, id=f'r15_{arm}_s{old["seed"]}_f{old["fold"]}', arm=arm, variant=arm,
                     grade_unit=unit['id'], grade_complete_sha256=core.digest(out/'complete.json'), **plan)
            key = effective_key(j)
            if key in seen:
                j['alias'] = seen[key]
            else:
                seen[key] = dict(kind='student', id=j['id'])
            jobs.append(j)
    assert len(jobs) == 30
    core.save_json(STUDY/'final_jobs.json', jobs)
    training = dict(round=15, seeds=SEEDS, variants=['full_ce']+ARMS, grading_config_sha256=core.digest(STUDY/'grading_config.json'),
                    jobs_sha256=core.digest(STUDY/'final_jobs.json'), grade_file_hashes=hashes,
                    pretrained_sha256=cfg['pretrained_sha256'], new_neural_trainings=sum('alias' not in j for j in jobs),
                    reused_baselines=15, aliases=[dict(id=j['id'], source=j['alias']) for j in jobs if 'alias' in j],
                    bootstrap=dict(groups=185, draws=20000, seed=20261024, family_percentiles=[1.25,98.75]))
    core.save_json(STUDY/'config.json', training)
    arch.commit('research: freeze fit-only nested grades and matched student weights/subsets before training')


def validate_predictions(path, ids, records):
    rows = sorted(arch.read_csv(path), key=lambda r: int(r['index']))
    assert [int(r['index']) for r in rows] == sorted(ids)
    p = np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])
    y = np.array([int(r['label']) for r in rows])
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    assert np.array_equal(p.argmax(1), [int(r['pred']) for r in rows])
    for r in rows:
        m = records[int(r['index'])]
        assert m['split'] == 'train' and m['sample_id'] == r['sample_id'] and m['group_id'] == r['group_id'] and m['label'] == int(r['label'])
    return rows, core.metrics(p,y)


def student_done(job):
    path = STUDY/'runs'/job['id']/'outer_complete.json'
    if not path.exists():
        return False
    value = core.read_json(path)
    assert value['signature'] == core.signature(job) and value['n'] == len(job['score'])
    assert core.digest(path.parent/'oof_predictions.csv') == value['prediction_sha256']
    if 'alias' not in job:
        assert core.complete(job)
    else:
        source = STUDY/('baselines' if job['alias']['kind'] == 'baseline' else 'runs')/job['alias']['id']/'complete.json'
        assert core.digest(source) == value['alias_source_complete_sha256']
        assert core.digest(ROOT/value['checkpoint']) == value['checkpoint_sha256']
    return True


def worker(job_id):
    if job_id.startswith('grade_'):
        verify_grading()
        grade_worker(next(j for j in core.read_json(STUDY/'grading_jobs.json') if j['id'] == job_id))
        return
    verify()
    job = next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id'] == job_id)
    if student_done(job):
        return
    out = STUDY/'runs'/job_id
    out.mkdir(parents=True, exist_ok=True)
    extra = {}
    if 'alias' in job:
        source = STUDY/('baselines' if job['alias']['kind'] == 'baseline' else 'runs')/job['alias']['id']
        comp = core.read_json(source/'complete.json')
        shutil.copyfile(source/'oof_predictions.csv', out/'oof_predictions.csv')
        extra['alias_source_complete_sha256'] = core.digest(source/'complete.json')
        core.save_json(out/'alias.json', dict(signature=core.signature(job), source=job['alias'], **extra))
    else:
        if not core.complete(job):
            core.run_worker(job_id)
        comp = core.read_json(out/'complete.json')
    records = core.read_json(STUDY/'manifest.json')
    rows, metrics = validate_predictions(out/'oof_predictions.csv', job['score'], records)
    assert abs(metrics['macro_f1']-comp['outer_metrics']['macro_f1']) < 1e-12
    core.save_json(out/'outer_complete.json', dict(signature=core.signature(job), n=len(rows),
                   prediction_sha256=core.digest(out/'oof_predictions.csv'), checkpoint=comp['checkpoint'],
                   checkpoint_sha256=comp['checkpoint_sha256'], **extra, **metrics))


def check():
    import torch
    import torch.nn.functional as F
    cfg = verify_grading()
    features = load_features()
    records = core.read_json(STUDY/'manifest.json')
    jobs = core.read_json(STUDY/'grading_jobs.json')
    assert core.digest(core.CACHE/'resized_uint8.npy') == core.read_json(core.CACHE/'meta.json')['cache_sha256']
    coverage = collections.Counter()
    for job in jobs:
        frozen_fold = next(f for f in core.read_json(SOURCE/'folds.json') if f['seed']==job['seed'] and f['fold']==job['fold'])
        for key in ['train','select','score']: assert job[key]==frozen_fold[key]
        core.validate_split(records, job['train'], job['select'], job['score'])
        assert all(0 <= i < 2386 for key in ['train','select','score'] for i in job[key])
        coverage.update((job['seed'],i) for i in job['score'])
        base = STUDY/'baselines'/job['source_job_id']
        assert [int(r['epoch']) for r in arch.read_csv(base/'train_log.csv')] == list(range(60))
        old = next(e['source_job'] for e in core.read_json(STUDY/'baseline_index.json') if e['source_job']['id'] == job['source_job_id'])
        comp = core.read_json(base/'complete.json')
        assert comp['signature'] == original_signature(old)
        assert core.digest(ROOT/comp['checkpoint']) == comp['checkpoint_sha256']
        _, metric = validate_predictions(base/'oof_predictions.csv', old['score'], records)
        assert abs(metric['macro_f1']-comp['outer_metrics']['macro_f1']) < 1e-12
    assert len(coverage) == 2386*3 and set(coverage.values()) == {1}
    job = jobs[0]
    grade_worker(job)
    assert grade_done(job)
    labels = np.array([r['label'] for r in records]); groups = np.array([r['group_id'] for r in records])
    manifest = core.read_json(STUDY/'grades'/job['id']/'head_manifest.json')
    for h in manifest['heads'].values():
        with np.load(ROOT/h['path'], allow_pickle=False) as data:
            fit, held = data['fit_indices'], data['held_indices']
            assert set(fit) | set(held) == set(job['train'])
            assert set(groups[fit]).isdisjoint(groups[held])
            assert set(groups[fit]) | set(groups[held]) == set(groups[job['train']])
            assert np.array_equal(data['mean'], features[fit].mean(0))
            assert np.array_equal(data['scale'], np.maximum(features[fit].std(0),1e-6))
            x = (features[fit]-data['mean'])/data['scale']
            p = grader.softmax(x@data['coef'].T+data['intercept'])
            residual = p-np.eye(3)[labels[fit]]
            assert np.max(np.abs(x.T@residual/len(fit)+data['coef'].T/len(fit))) < 5e-5
    plans = core.read_json(STUDY/'grades'/job['id']/'student_plans.json')
    for label in [0,1,2]:
        ids = [i for i in job['train'] if labels[i] == label]
        a = [plans['grade_weighted']['sample_weights'][str(i)] for i in ids]
        b = [plans['shuffled_weighted']['sample_weights'][str(i)] for i in ids]
        assert sorted(a) == sorted(b) and abs(np.mean(a)-1) < 1e-12
        assert sum(labels[i] == label for i in plans['ac_only']['train']) == sum(labels[i] == label for i in plans['random_ac']['train'])
    x,y,_,idx = next(iter(core.loader(plans['ac_only']['train'][:32], False, seed=SEEDS[0])))
    a,b = arch.ORIGINAL_MODEL(SEEDS[0]).eval(), arch.ORIGINAL_MODEL(SEEDS[0]).eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        assert torch.equal(a(x.cuda()), b(x.cuda()))
    del b
    a.train(); optimizer = torch.optim.AdamW(a.parameters(), lr=1e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    weights = torch.ones(len(y), device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss = (F.cross_entropy(a(x.cuda()), y.cuda(), reduction='none')*weights).mean()
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    assert a.fc.weight.grad.abs().sum() > 0
    scaler.step(optimizer); scaler.update()
    core.save_json(STUDY/'preflight_validation.json', dict(passed=True, timestamp=core.timestamp(), reused_grade_unit=job['id'],
                   checks=['15 signed baseline checkpoints, 60 epochs and full outer predictions', 'exact original folds and train-only coverage',
                           'full input and feature SHA, frozen public DINO columns', '15 converged fit-only linear heads and independent optimality residual',
                           'patient exclusions, per-class shuffled weight mass and matched AC/random counts',
                           'paired student initial logits and finite real AC-image unweighted AMP update; formal training keeps original dynamic scaler']))
    arch.commit('test: validate strict nested linear grading and matched identity controls')
    print(json.dumps(dict(passed=True, grader_units=15, linear_fits=225, new_students_max=30, reused_baselines=15)), flush=True)


def summarize():
    cfg = verify(); records = core.read_json(STUDY/'manifest.json')
    units = core.read_json(STUDY/'grading_jobs.json'); jobs = core.read_json(STUDY/'final_jobs.json')
    predictions = {}; results = []; folds = []
    for arm in ['full_ce']+ARMS:
        for seed in SEEDS:
            merged = []
            for unit in [u for u in units if u['seed'] == seed]:
                if arm == 'full_ce':
                    out = STUDY/'baselines'/unit['source_job_id']
                else:
                    job = next(j for j in jobs if j['seed'] == seed and j['fold'] == unit['fold'] and j['arm'] == arm)
                    assert student_done(job)
                    out = STUDY/'runs'/job['id']
                rows, metrics = validate_predictions(out/'oof_predictions.csv', unit['score'], records)
                merged += rows
                folds.append(dict(variant=arm, seed=seed, fold=unit['fold'], **metrics))
            merged.sort(key=lambda r:int(r['index']))
            assert [int(r['index']) for r in merged] == list(range(2386))
            predictions[arm,seed] = merged
            core.save_csv(STUDY/'oof'/f'{arm}_s{seed}.csv', merged)
            p = np.array([[float(r[f'p{k}']) for k in range(3)] for r in merged]); y = np.array([int(r['label']) for r in merged])
            metrics = core.metrics(p,y); groups = sorted({r['group_id'] for r in merged}); cm = np.array(metrics['confusion_matrix'])
            acc = np.mean([np.mean([int(r['pred']) == int(r['label']) for r in merged if r['group_id'] == g]) for g in groups])
            results.append(dict(variant=arm, seed=seed, n=2386, patient_groups=len(groups),
                           malignant_recall=float(cm[1,1]/cm[1].sum()), patient_mean_accuracy=float(acc), **metrics))
    draws = np.random.default_rng(20261024).multinomial(len(groups), np.ones(len(groups))/len(groups), size=20000)
    def matrix(rows):
        m = np.zeros((len(groups),3,3))
        np.add.at(m, ([groups.index(r['group_id']) for r in rows], [int(r['label']) for r in rows], [int(r['pred']) for r in rows]),1)
        return m
    def f1(m):
        d = m.sum(-1)+m.sum(-2)
        return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
    comparisons = []
    for candidate, reference in COMPARISONS:
        ds = []; boot = []
        for seed in SEEDS:
            a,b = matrix(predictions[candidate,seed]), matrix(predictions[reference,seed])
            ds.append(float(f1(a.sum(0))-f1(b.sum(0))))
            boot.append(f1(np.einsum('bg,gij->bij',draws,a))-f1(np.einsum('bg,gij->bij',draws,b)))
        interval = np.percentile(np.mean(boot,axis=0),[1.25,98.75]).tolist()
        comparisons.append(dict(candidate=candidate, reference=reference, seed_deltas=ds, mean_delta=float(np.mean(ds)),
                            patient_cluster95=np.percentile(np.mean(boot,axis=0),[2.5,97.5]).tolist(), family_interval=interval,
                            strong=bool(min(ds)>0 and np.mean(ds)>=.01 and interval[0]>0)))
    gates = dict(comparisons=comparisons, strong_primary_followup=all(c['strong'] for c in comparisons[:2]),
                 resamples=20000, groups=185)
    core.save_json(STUDY/'round15_results.json', results); core.save_json(STUDY/'fold_results.json', folds)
    core.save_json(STUDY/'paired_comparisons.json', gates)
    lines = ['# Round15: new-split confirmation of nested AC selection', '',
             '225 linear grader fits; up-to30 new CNN students and15 reused baselines. Only original-train patients. No outer filtering.', '',
             '| Pipeline | Seed | Pooled OOF F1 | Malignant recall | Patient mean accuracy |','|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["variant"]} | {r["seed"]} | {r["macro_f1"]:.6f} | {r["malignant_recall"]:.6f} | {r["patient_mean_accuracy"]:.6f} |')
    lines += ['', json.dumps(gates,indent=2), '', 'Grades are consistency proxies, not label-error truth. Internal conditional intervals do not cover all repeated development or overlapping-training uncertainty.']
    (STUDY/'ROUND15_REPORT.md').write_text('\n'.join(lines))
    arch.commit('results: replicate nested AC selection against full CE and count-matched random filtering')
    core.save_json(STUDY/'status.json',dict(phase='round15_complete_needs_analysis', round=15, completed=30,
                   new_neural_trainings=cfg['new_neural_trainings'], updated_at=core.timestamp()))


def run():
    import fcntl
    global STAGE
    with open(STUDY/'pipeline.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_grading()
            assert core.read_json(STUDY/'preflight_validation.json')['passed']
            arch.done = grade_done
            arch.run_jobs(core.read_json(STUDY/'grading_jobs.json'))
            freeze_training()
            STAGE = 'student_training'
            arch.done = student_done
            jobs = core.read_json(STUDY/'final_jobs.json')
            arch.run_jobs([j for j in jobs if 'alias' not in j])
            for j in jobs:
                if 'alias' in j: worker(j['id'])
            summarize()
        except Exception as error:
            core.save_json(STUDY/'status.json',dict(phase='failed',round=15,error=repr(error),updated_at=core.timestamp()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','worker','run'])
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'check': check()
    elif args.command == 'worker': worker(args.job)
    else: run()

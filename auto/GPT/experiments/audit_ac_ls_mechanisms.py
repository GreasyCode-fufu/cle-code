"""Independent checkpoint/inference audit and prespecified AC LS replication analysis."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import run_ac_ls_mechanisms as study
from models.optimize.cle_token_adapter import counted_flops

core = study.core
s = study.STUDY
cfg = study.verify()
cohort = core.read_json(study.COHORT / 'dataset_config.json')
records = core.read_json(s / 'manifest.json')
jobs = core.read_json(s / 'final_jobs.json')
assert len(jobs) == 15 and set(cfg['seeds']) == {45, 46, 47}
assert all(study.done(j) for j in jobs), 'Wait for all frozen jobs; never train in audit.'
assert core.digest(core.PRETRAINED) == cfg['pretrained_sha256']
assert core.digest(core.CACHE / 'resized_uint8.npy') == core.read_json(core.CACHE / 'meta.json')['cache_sha256']
assert core.read_json(study.COHORT / 'independent_dataset_audit.json')['passed']
assert [len(cohort['indices'][k]) for k in ['train', 'val', 'test']] == [2077, 427, 576]
groups = sorted({records[i]['group_id'] for i in cohort['indices']['val']})
assert len(groups) == 39
group_index = {name: i for i, name in enumerate(groups)}


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def signature(job, config):
    return hashlib.sha256((json.dumps(job, sort_keys=True) + core.digest(config) + core.digest(core.__file__)).encode()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_predictions(path, indices):
    rr = rows(path)
    assert [int(row['index']) for row in rr] == indices
    assert len(rr) == len(set(indices)) == 427
    p = np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    y = np.array([int(row['label']) for row in rr])
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    assert np.array_equal(p.argmax(1), [int(row['pred']) for row in rr])
    for row in rr:
        rec = records[int(row['index'])]
        assert row['sample_id'] == rec['sample_id'] and row['group_id'] == rec['group_id'] and int(row['label']) == rec['label']
    matrix = np.zeros((39, 3, 3), dtype=np.float64)
    np.add.at(matrix, ([group_index[row['group_id']] for row in rr], y, p.argmax(1)), 1)
    return p, y, matrix


def metric_from_cm(matrix):
    den = matrix.sum(-1) + matrix.sum(-2)
    return np.mean(np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), den,
                             out=np.zeros_like(den), where=den != 0), axis=-1)


all_jobs = [(j, s / 'runs' / j['id'], j['variant'], s / 'config.json') for j in jobs]
models = []
curves = []
predictions = {}
errors = collections.defaultdict(lambda: dict(wrong_votes=0, workflows=set()))
budgets = {b['arm']: b for b in core.read_json(s / 'preflight_validation.json')['budgets']}
for job, out, variant, config in all_jobs:
    seed = job['seed']
    assert job['train'] == cohort['indices']['train'] and not job['sample_weights']
    assert job['select'] == job['score'] == cohort['indices']['val']
    assert not set(job['train']) & set(job['score'])
    assert not {records[i]['group_id'] for i in job['train']} & {records[i]['group_id'] for i in job['score']}
    assert {records[i]['split'] for i in job['train']} == {'train'}
    assert {records[i]['split'] for i in job['score']} == {'val'}
    assert not set(job['train'] + job['score']) & set(cohort['indices']['test'])
    comp = core.read_json(out / 'complete.json')
    meta = core.read_json(out / 'run_meta.json')
    result = core.read_json(out / 'validation_complete.json')
    expected_signature = signature(job, config)
    assert comp['signature'] == meta['signature'] == result['signature'] == expected_signature
    assert comp['train_count'] == meta['train_count'] == 2077 and comp['select_count'] == 427
    assert comp['pretrained_sha256'] == meta['pretrained_sha256'] == cfg['pretrained_sha256']
    assert comp['checkpoint_sha256'] == core.digest(ROOT / comp['checkpoint']) and comp['epochs_completed'] == 60
    log = rows(out / 'train_log.csv')
    assert [int(r['epoch']) for r in log] == list(range(60))
    assert all(np.isfinite(float(v)) for row in log for v in row.values()) and float(log[-1]['lr']) == 0
    best = int(np.argmax([float(r['val_macro_f1']) for r in log]))
    assert best == comp['best_epoch'] and abs(float(log[best]['val_macro_f1']) - comp['best_select_f1']) < 1e-12
    saved, y, matrix = validate_predictions(out / 'validation_predictions.csv', job['score'])
    m = core.metrics(saved, y)
    assert abs(m['macro_f1'] - result['macro_f1']) < 1e-12
    assert abs(m['macro_f1'] - comp['best_select_f1']) < 1e-12
    assert m['confusion_matrix'] == result['confusion_matrix']
    assert comp['checkpoint_sha256'] == result['checkpoint_sha256']
    for name, sha in result['file_hashes'].items():
        assert core.digest(out / name) == sha
    model = study.model_for(seed, variant, job['control_width'])
    initial_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not name.startswith('modifier.')}
    initial_bn = {name: int(v) for name, v in model.state_dict().items() if name.endswith('num_batches_tracked')}
    state = torch.load(ROOT / comp['checkpoint'], map_location='cpu', weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state, strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == parameters
    assert parameters == result['parameters'] == budgets[variant]['parameters']
    assert sum(p.numel() for p in model.modifier.parameters()) == result['modifier_parameters'] == budgets[variant]['modifier_parameters']
    if variant == 'plain':
        assert parameters == 21286211 and model.modifier.branch is None
    for name, initial in initial_bn.items():
        assert int(state[name]) == initial + 65 * (best + 1), (job['id'], name, int(state[name]), initial, best)
    squared = sum(float((p.detach().cpu().double() - initial_params[name].double()).square().sum())
                  for name, p in model.named_parameters() if name in initial_params)
    count = sum(p.numel() for p in initial_params.values())
    drift = (squared / count) ** .5
    assert drift > 0
    del initial_params
    model.eval()
    state_before = state_sha(model)
    changes = []
    features = []
    if variant.startswith('ls_'):
        def capture(module, args):
            if not features:
                features.append(args[0].detach())
        kernel_hook = model.modifier.branch.register_forward_pre_hook(capture)
    def hook(module, inputs, output):
        value = (output - inputs[0]).abs().flatten(1).mean(1) / (inputs[0].abs().flatten(1).mean(1) + 1e-8)
        changes.extend(value.detach().cpu().float().tolist())
    handle = model.modifier.register_forward_hook(hook)
    p, yr, idx = core.infer(model, core.loader(job['score'], False, seed=seed))
    assert idx == job['score'] and np.array_equal(yr, y)
    diff = float(np.max(np.abs(saved - p)))
    assert diff < 2e-6 and np.array_equal(saved.argmax(1), p.argmax(1))
    diag = None
    handle.remove()
    if variant.startswith('ls_'):
        kernel_hook.remove()
        branch = model.modifier.branch
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            raw = branch.predictor(features[0], positive=False)
            actual = branch.weights(features[0])
            expected = raw.softmax(2) if variant == 'ls_positive' else raw
            if variant == 'ls_global':
                expected = raw.mean((-2, -1), keepdim=True).expand_as(raw)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected)
            if variant == 'ls_positive':
                assert (actual >= 0).all() and torch.allclose(actual.sum(2), torch.ones_like(actual[:, :, 0]), atol=.002)
            if variant == 'ls_global':
                assert torch.equal(actual, actual[:, :, :, :1, :1].expand_as(actual))
        del raw, actual, expected
    if variant == 'plain':
        assert max(changes) == 0
    assert len(changes) == 427 and np.isfinite(changes).all()
    diag = result['diagnostics']
    assert abs(float(np.mean(changes)) - diag['stream_change_mean']) < 2e-6
    off_saved, off_y, _ = validate_predictions(out / 'off_predictions.csv', job['score'])
    model.modifier.enabled = False
    off, yo, io = core.infer(model, core.loader(job['score'], False, seed=seed))
    assert io == idx and np.array_equal(yo, y) and np.array_equal(off_y, y)
    assert np.max(np.abs(off - off_saved)) < 2e-6 and np.array_equal(off.argmax(1), off_saved.argmax(1))
    assert core.metrics(off, yo)['confusion_matrix'] == diag['off_metrics']['confusion_matrix']
    assert abs(float(np.mean(p.argmax(1) != off.argmax(1))) - diag['off_prediction_disagreement']) < 1e-12
    assert state_sha(model) == state_before
    predictions[variant, seed] = matrix
    for i, yi, pi in zip(idx, y, p.argmax(1)):
        if yi != pi:
            errors[i]['wrong_votes'] += 1
            errors[i]['workflows'].add(variant)
    cm = np.array(m['confusion_matrix'])
    record = dict(job=job['id'], variant=variant, seed=seed, checkpoint_sha256=comp['checkpoint_sha256'],
                  signature=expected_signature, epochs=60, best_epoch=best, macro_f1=m['macro_f1'], accuracy=m['accuracy'],
                  malignant_recall=float(cm[1, 1] / cm[1].sum()),
                  patient_equal_accuracy=float(np.mean(np.trace(matrix, axis1=-2, axis2=-1) / matrix.sum((1, 2)))),
                  confusion_matrix=m['confusion_matrix'], parameters=parameters, source_parameter_rms_drift=drift,
                  probability_max_difference=diff, inference_state_unchanged=True, bn_buffers_verified=len(initial_bn), diagnostics=diag)
    models.append(record)
    curves.append(dict(variant=variant, seed=seed, best_epoch=best, best_val_f1=m['macro_f1'],
                       final_val_f1=float(log[-1]['val_macro_f1']), final_train_accuracy=float(log[-1]['train_acc']),
                       final_train_loss=float(log[-1]['train_loss']), history=log))
    assert (ROOT / comp['checkpoint']).owner() == 'login'
    assert all(p.owner() == 'login' for p in out.rglob('*') if p.is_file())
    print('audited', job['id'], flush=True)
    del model, state, features
    torch.cuda.empty_cache()

weights = np.random.default_rng(20261024).multinomial(39, np.ones(39) / 39, size=20000)
comparisons = []
for control in cfg['primary_references'] + cfg['mechanism_references']:
    ds = []
    boots = []
    for seed in cfg['seeds']:
        candidate, reference = predictions[cfg['primary'], seed], predictions[control, seed]
        ds.append(float(metric_from_cm(candidate.sum(0)) - metric_from_cm(reference.sum(0))))
        boots.append(metric_from_cm(np.einsum('bg,gij->bij', weights, candidate)) -
                     metric_from_cm(np.einsum('bg,gij->bij', weights, reference)))
    distribution = np.mean(boots, axis=0)
    comparisons.append(dict(candidate=cfg['primary'], reference=control, seed_deltas=ds, mean_delta=float(np.mean(ds)),
                            patient95=np.percentile(distribution, [2.5, 97.5]).tolist(),
                            family_interval=np.percentile(distribution, [.625, 99.375]).tolist(),
                            family_percentiles=[.625, 99.375], patients=39, bootstrap_replicates=20000))

for comparison in comparisons:
    comparison['role'] = 'primary' if comparison['reference'] in cfg['primary_references'] else 'mechanism'
    comparison['family_lower_positive'] = bool(comparison['family_interval'][0] > 0)
    if comparison['role'] == 'primary':
        comparison['primary_pair_passed'] = bool(min(comparison['seed_deltas']) > 0 and
            comparison['mean_delta'] >= cfg['primary_minimum_delta'] and comparison['family_interval'][0] > 0)
primary_passed = all(r['primary_pair_passed'] for r in comparisons if r['role'] == 'primary')
assert len([r for r in comparisons if r['role'] == 'primary']) == 2
# Check the matrix implementation independently using sklearn weighted-label F1.
from sklearn.metrics import f1_score
truth, predicted = np.indices((3, 3)).reshape(2, -1)
for matrix in predictions.values():
    for draw in weights[:8]:
        cm = np.einsum('g,gij->ij', draw, matrix)
        independent = f1_score(truth, predicted, sample_weight=cm.ravel(), labels=[0, 1, 2], average='macro', zero_division=0)
        assert abs(independent - float(metric_from_cm(cm))) < 1e-12

summary = []
for variant in study.ARMS:
    rr = [r for r in models if r['variant'] == variant]
    assert len(rr) == 3
    summary.append(dict(variant=variant, seed_f1=[r['macro_f1'] for r in rr],
                        mean_f1=float(np.mean([r['macro_f1'] for r in rr])), seed_sd=float(np.std([r['macro_f1'] for r in rr], ddof=1)),
                        mean_malignant_recall=float(np.mean([r['malignant_recall'] for r in rr])),
                        mean_patient_accuracy=float(np.mean([r['patient_equal_accuracy'] for r in rr])), parameters=rr[0]['parameters']))

resource_rows = rows(s / 'resource_samples.csv')
steady = [r for r in resource_rows[15:] if int(r['running']) >= 6]
resource_summary = dict(samples=len(resource_rows), full_six_job_samples=len(steady),
                        mean_gpu_percent=float(np.mean([float(r['gpu_util']) for r in steady])),
                        mean_cpu_percent=float(np.mean([float(r['cpu_percent']) for r in steady])),
                        mean_iowait_percent=float(np.mean([float(r['cpu_iowait_percent']) for r in steady])),
                        gpu_memory_max_mib=max(int(r['gpu_used_mib']) for r in resource_rows))
error_rows = [dict(index=i, sample_id=records[i]['sample_id'], group_id=records[i]['group_id'], label=records[i]['label'],
                   wrong_votes=value['wrong_votes'], workflows=sorted(value['workflows'])) for i, value in errors.items()]
error_rows.sort(key=lambda r: (-r['wrong_votes'], r['index']))
audit = dict(passed=True, timestamp=core.timestamp(), new_neural_models=15, reused_models=0, train_images=2077,
             val_images=427, test_model_evaluations=0, models=models,
             checks=['frozen cohort, source/config/manifest/prior-audit hashes and all checkpoint signatures',
                     '60epoch logs and earliest highest validation checkpoint', 'same AC2077/427 lists and patient isolation, no test loader',
                     'all 15 checkpoint reloads/whole-val inference and 15 branch-off inferences including identity plain',
                     'finite parameters and exact budgets, BN counts and pretrained parameter drift',
                     'per-image probabilities/metrics/diagnostics and unchanged inference state',
                     'kernel laws independently reconstructed from best-model real features',
                     'four prespecified 39-patient shared bootstrap comparisons and co-primary gate',
                     'independent sklearn weighted-label verification of bootstrap F1', 'login ownership'])
core.save_json(s / 'independent_audit.json', audit)
core.save_json(s / 'round2_results.json', models)
core.save_json(s / 'round2_summary.json', summary)
core.save_json(s / 'paired_comparisons.json', dict(comparisons=comparisons, strong_internal_replication=primary_passed))
core.save_json(s / 'training_curves.json', curves)
core.save_json(s / 'error_distribution.json', dict(val_errors_only=True, rows=error_rows, note='Do not change frozen grades or labels based on these errors.'))
core.save_json(s / 'resource_summary.json', resource_summary)
lines = ['# A–C第二轮：LS新种子复核与机制对照', '',
         '15项新训练，固定ACtrain2077/val427、seed45/46/47，未评价ACtest576。新seed不是新患者。', '',
         '| 流程 | 三seed宏F1 | 平均宏F1 | 恶性召回均值 | 患者等权准确率 | 参数 |',
         '|---|---|---:|---:|---:|---:|']
for r in summary:
    lines.append(f'| {r["variant"]} | {r["seed_f1"]} | {r["mean_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_accuracy"]:.6f} | {r["parameters"]} |')
lines += ['', '四预定比较，39患者20000共享bootstrap；差值为百分点。', '',
          '| LS相对 | 角色 | 三seed差 | 均值差 | 普通95% | 四比较家族区间 |', '|---|---|---|---:|---|---|']
for r in comparisons:
    lines.append(f'| {r["reference"]} | {r["role"]} | {[round(100*x,4) for x in r["seed_deltas"]]} | {100*r["mean_delta"]:+.4f} | {[round(100*x,4) for x in r["patient95"]]} | {[round(100*x,4) for x in r["family_interval"]]} |')
lines += ['', f'共同主要门槛是否全部通过：{primary_passed}。两项都须三个seed差正、均值至少+.5百分点、四家族下界正；机制比较不替代主要比较，新判据不回改首轮。',
          '独立重建15最佳模型全部val on/off推理，核验签名/SHA/60ep/最早最佳/普通BN计数/源漂移/有限权重/预算/所有权，并从真实特征重算三种核法。',
          '卷积92672与LS94432新增参数差1.86%；三种LS相同预算。Conv/Linear计数不含所有算子，不能声称严格同计算或官方LSNet整网性能。',
          'val也用于选择checkpoint，同患者反复开发/初始化/选择的不确定性未被条件区间全覆盖，不称独立临床确认或新颖网络。等级不是医生真值；test分级已用过其现有标签。',
          '同checkpoint禁分支是诊断，不能替代正核/全局核的重训对照。所有阴性与错误保留，不因新模型改AC清单或原标签。候选最终冻结后统一评价ACtest576。',
          f'六任务稳定窗口GPU均值{resource_summary["mean_gpu_percent"]:.2f}%，CPU{resource_summary["mean_cpu_percent"]:.2f}%，iowait{resource_summary["mean_iowait_percent"]:.4f}%；不把准备分析阶段算为持续满载。']
(s / 'ROUND2_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
assert all(p.owner() == 'login' for p in s.rglob('*') if p.is_file())
subprocess.run([core.GIT, '-C', str(ROOT), 'add', 'experiments/audit_ac_ls_mechanisms.py', str(s.relative_to(ROOT))], check=True)
subprocess.run([core.GIT, '-C', str(ROOT), 'diff', '--cached', '--check'], check=True)
subprocess.run([core.GIT, '-C', str(ROOT), 'commit', '-m', 'results: independently audit AC LS replication and kernel-law controls'], check=True)
core.save_json(s / 'status.json', dict(phase='independently_audited', updated_at=core.timestamp(), completed=15, strong_internal_replication=primary_passed))
print(json.dumps(dict(passed=True, summary=summary, comparisons=comparisons, strong_internal_replication=primary_passed, resource=resource_summary)), flush=True)

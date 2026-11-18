"""Post-clarification CPU re-audit of completed predictions, no new inference."""
from pathlib import Path
from fractions import Fraction
import ast
import csv
import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import traceback
import numpy as np
import sklearn
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score)

S = Path(__file__).resolve().parent
R = S.parents[1]
NAMES = ['Benign_lesions', 'Malignant_lesions', 'body_of_stomach_normal']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def save(name, obj):
    def clean(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [clean(v) for v in x]
        return x
    target = S/name; target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(target)


def flatten(value):
    result = {k: float(v) for k, v in value.items() if isinstance(v, (float, int, np.number))}
    for c, name in enumerate(NAMES):
        for k in ['precision', 'recall', 'sensitivity', 'specificity', 'f1', 'npv']:
            result[f'class{c}.{k}'] = float(value['per_class'][name][k])
        for source, suffix in [('auc_roc_per_class', 'auroc'), ('auc_pr_per_class', 'auc_pr')]:
            result[f'class{c}.{suffix}'] = float(value[source][name])
    return result


def exact_cm(cm):
    n = int(cm.sum()); row = cm.sum(1); col = cm.sum(0)
    def ratio(a, b): return Fraction(int(a), int(b)) if b else Fraction()
    tp = [int(cm[k, k]) for k in range(3)]
    precision = [ratio(tp[k], col[k]) for k in range(3)]
    recall = [ratio(tp[k], row[k]) for k in range(3)]
    f1 = [ratio(2*tp[k], row[k]+col[k]) for k in range(3)]
    specificity = [ratio(n-row[k]-col[k]+tp[k], n-row[k]) for k in range(3)]
    npv = [ratio(n-row[k]-col[k]+tp[k], n-col[k]) for k in range(3)]
    out = dict(overall_accuracy=ratio(sum(tp), n), macro_f1=sum(f1)/3,
               macro_precision=sum(precision)/3, macro_recall=sum(recall)/3,
               macro_sensitivity=sum(recall)/3, balanced_accuracy=sum(recall)/3,
               macro_specificity=sum(specificity)/3,
               weighted_precision=sum(precision[k]*int(row[k]) for k in range(3))/n,
               weighted_recall=sum(recall[k]*int(row[k]) for k in range(3))/n,
               weighted_f1=sum(f1[k]*int(row[k]) for k in range(3))/n)
    chance = sum(int(row[k])*int(col[k]) for k in range(3))
    out['kappa'] = ratio(n*sum(tp)-chance, n*n-chance)
    for c in range(3):
        for k, values in [('precision', precision), ('recall', recall), ('sensitivity', recall),
                          ('specificity', specificity), ('f1', f1), ('npv', npv)]:
            out[f'class{c}.{k}'] = values[c]
    return out


def exact_rank(prob, y):
    aucs = []; aps = []; out = {}
    for c in range(3):
        score = prob[:, c]; positive = y == c
        pairs = score[positive, None]-score[None, ~positive]
        auc = Fraction(2*int((pairs > 0).sum())+int((pairs == 0).sum()), 2*pairs.size)
        order = np.argsort(-score, kind='stable'); labels = positive[order]
        ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score)-1]
        counts = np.cumsum(labels)[ends]; previous = 0; ap = Fraction()
        for endpoint, count in zip(ends, counts):
            ap += Fraction(int(count-previous), int(positive.sum()))*Fraction(int(count), int(endpoint+1)); previous = count
        out[f'class{c}.auroc'] = auc; out[f'class{c}.auc_pr'] = ap
        aucs.append(auc); aps.append(ap)
    out['auc_roc_macro'] = sum(aucs)/3; out['auc_pr_macro'] = sum(aps)/3
    return out


def bootstrap_cm(probabilities, y, groups, weights):
    unique = sorted(set(groups))
    per_patient = np.array([[[np.sum((y[groups == g] == a) & (p.argmax(1)[groups == g] == b))
                              for b in range(3)] for a in range(3)]
                            for p in probabilities for g in unique]).reshape(3, len(unique), 3, 3)
    cm = np.einsum('dg,sgij->dsij', weights, per_patient, optimize=True)
    tp = np.diagonal(cm, axis1=-2, axis2=-1); row = cm.sum(-1); col = cm.sum(-2); n = row.sum(-1)
    def div(a, b): return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    prec = div(tp, col); rec = div(tp, row); f1 = div(2*tp, row+col)
    spec = div(n[..., None]-row-col+tp, n[..., None]-row)
    npv = div(n[..., None]-row-col+tp, n[..., None]-col)
    out = dict(overall_accuracy=div(tp.sum(-1), n), macro_f1=f1.mean(-1),
               macro_precision=prec.mean(-1), macro_recall=rec.mean(-1),
               macro_sensitivity=rec.mean(-1), balanced_accuracy=rec.mean(-1),
               macro_specificity=spec.mean(-1), weighted_precision=div((prec*row).sum(-1), n),
               weighted_recall=div((rec*row).sum(-1), n), weighted_f1=div((f1*row).sum(-1), n))
    for c in range(3):
        for k, values in [('precision', prec), ('recall', rec), ('sensitivity', rec),
                          ('specificity', spec), ('f1', f1), ('npv', npv)]:
            out[f'class{c}.{k}'] = values[..., c]
    return {k: value.mean(1) for k, value in out.items()}


def bootstrap_rank(probabilities, y, groups, weights):
    ids = np.array([sorted(set(groups)).index(g) for g in groups]); store = {}
    for c in range(3):
        auc_seeds = []; ap_seeds = []
        for p in probabilities:
            score = p[:, c]; order = np.argsort(-score, kind='stable'); positive = (y[order] == c)
            ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y)-1]
            auc_values = []; ap_values = []
            for start in range(0, len(weights), 256):
                w = weights[start:start+256, ids[order]]
                tp = np.cumsum(w*positive, axis=1)[:, ends]
                fp = np.cumsum(w*(~positive), axis=1)[:, ends]
                dtp = np.diff(np.c_[np.zeros(len(w)), tp], axis=1)
                dfp = np.diff(np.c_[np.zeros(len(w)), fp], axis=1)
                prev_tp = np.c_[np.zeros(len(w)), tp[:, :-1]]
                den = tp[:, -1]*fp[:, -1]
                auc_values.extend(np.divide((dfp*(tp+prev_tp)/2).sum(1), den, out=np.full(len(w),np.nan), where=den!=0).tolist())
                precision = np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp!=0)
                ap_values.extend(np.divide((precision*dtp).sum(1),tp[:, -1],out=np.full(len(w),np.nan),where=tp[:, -1]!=0).tolist())
            auc_seeds.append(auc_values); ap_seeds.append(ap_values)
        store[f'class{c}.auroc'] = np.array(auc_seeds).mean(0)
        store[f'class{c}.auc_pr'] = np.array(ap_seeds).mean(0)
    store['auc_roc_macro'] = np.mean([store[f'class{c}.auroc'] for c in range(3)], axis=0)
    store['auc_pr_macro'] = np.mean([store[f'class{c}.auc_pr'] for c in range(3)], axis=0)
    assert all(np.isfinite(x).sum() >= .99*len(x) for x in store.values())
    return store


def main():
    assert os.getuid() == 1000 and socket.gethostname() == 'Dell-T2' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cfg = read(S/'config.json'); assert cfg['source_sha256'] == sha(__file__)
    assert cfg['policy_sha256']==sha(S/'metric_scope_policy.json') and cfg['protocol_sha256']==sha(S/'PROTOCOL_ZH.md')
    assert all(sha(R/p)==h for p,h in cfg['locked_authority'].items())
    reference = S/'evaluate_reference.py'
    assert sha(reference) == cfg['evaluate_sha256'] == sha(R/'evaluate.py')
    node = next(n for n in ast.parse(reference.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'compute_metrics')
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.Name) and n.id in ['open', 'exec', 'eval', '__import__', 'os', 'sys'] for n in ast.walk(node))
    namespace = dict(globals()); exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(reference), 'exec'), namespace)
    metric = namespace['compute_metrics']
    records = read(R/'experiments/ac_dataset_v1/dataset_manifest.json'); authority = {r['index']: r for r in records}
    ids = read(R/'experiments/ac_dataset_v1/dataset_config.json')['indices']['test']
    y = np.array([authority[i]['label'] for i in ids]); groups = np.array([authority[i]['group_id'] for i in ids])
    assert len(ids) == 576 and len(set(groups)) == 41
    weights = np.random.default_rng(20261024).multinomial(41, np.full(41, 1/41), size=20000)
    manifest = {}; registry = []; results = {}; data_cache = {}; cm_cache = {}; rank_cache = {}
    def preserve(path): manifest[str(path.relative_to(R))] = sha(path)
    def checked(path):
        preserve(path); rows = list(csv.DictReader(path.open())); prefix = '' if 'p0' in rows[0] else 'mean_'
        assert [int(v['index']) for v in rows] == ids
        assert all(int(v['label']) == authority[i]['label'] and v['group_id'] == authority[i]['group_id'] for i, v in zip(ids, rows))
        p = np.array([[float(v[f'{prefix}p{k}']) for k in range(3)] for v in rows])
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 and np.allclose(p.sum(1), 1, atol=1e-6)
        return p
    def add(name, paths, meta):
        ps = [checked(p) for p in paths]; scores = []; fractions = []
        for p in ps:
            value = metric(y, p.argmax(1), p, 3, NAMES); flat = flatten(value)
            exact = {**exact_cm(value['confusion_matrix']), **exact_rank(p, y)}
            assert all(abs(float(v)-flat[k]) < 1e-12 for k, v in exact.items()), name
            scores.append(flat); fractions.append({k: str(v) for k, v in exact.items()})
        results[name] = dict(**meta, metrics={k:dict(seeds=[v[k] for v in scores],mean=float(np.mean([v[k] for v in scores]))) for k in scores[0]}, exact_fractions=fractions, paths=[str(p.relative_to(R)) for p in paths])
        registry.append(dict(name=name, **meta)); data_cache[name] = ps
        cm_cache[name] = bootstrap_cm(ps, y, groups, weights)
    T = R/'experiments/ac_test_feedback_window01'; baseline = read(T/'methods_v2/qualification_convnext_tiny_A.json'); preserve(T/'methods_v2/qualification_convnext_tiny_A.json')
    add('B0_single', [T/'predictions'/u['fingerprint']/'test_predictions.csv' for u in baseline['units']], dict(role='fixed B0', mode='single', parameters=27822435, counted_flops=8909531136))
    add('B0_c4', [R/'experiments/ac_c4_budget_controls_window01/controls/B0/runs'/f's{s}'/'test_predictions.csv' for s in [45,46,47]], dict(role='same mature C4 budget B0', mode='c4', parameters=27822435, counted_flops=4*8909531136))
    comparisons = {}; snapshots = []
    def compare(candidate, reference_name):
        a = results[candidate]; b = results[reference_name]; values = {}
        rank_needed = any(k not in cm_cache[candidate] and k not in ['kappa', 'mcc'] and v['mean']-b['metrics'][k]['mean'] >= .015 and all(x>z for x,z in zip(v['seeds'],b['metrics'][k]['seeds'])) for k,v in a['metrics'].items())
        if rank_needed:
            for name in [candidate, reference_name]:
                if name not in rank_cache: rank_cache[name] = bootstrap_rank(data_cache[name], y, groups, weights)
        for key, av in a['metrics'].items():
            bv = b['metrics'][key]; ds = [x-z for x,z in zip(av['seeds'],bv['seeds'])]; mean = av['mean']-bv['mean']
            if key in cm_cache[candidate]: distribution = cm_cache[candidate][key]-cm_cache[reference_name][key]
            elif rank_needed and key in rank_cache[candidate]: distribution = rank_cache[candidate][key]-rank_cache[reference_name][key]
            else: distribution = None
            ci = np.nanpercentile(distribution,[2.5,97.5]).tolist() if distribution is not None else None
            exact_delta = None
            if key in a['exact_fractions'][0]:
                delta = sum((Fraction(x[key])-Fraction(z[key]) for x,z in zip(a['exact_fractions'],b['exact_fractions'])), Fraction())/3
                assert abs(float(delta)-mean) < 1e-12; exact_delta = str(delta)
            eligible = key not in ['kappa','mcc']
            values[key] = dict(mean_delta=mean, paired_seed_deltas=ds, exact_mean_fraction=exact_delta,
                conditional_ci95=ci, all_seeds_positive=all(x>0 for x in ds), at_least_1p5pp=eligible and mean>=.015,
                valid_bootstrap_draws=int(np.isfinite(distribution).sum()) if distribution is not None else None,
                conditional_numeric_qualified=bool(eligible and mean>=.015 and all(x>0 for x in ds) and ci is not None and ci[0]>0),
                CI_scope='paired41patient20k, same patient draws across seeds; not native evaluate.py image bootstrap; adaptive model/metric selection unadjusted',
                CI_missing_reason=None if ci is not None else ('coefficient supplementary' if not eligible else 'rank metric CI computed only for eligible >=1.5pp allseed-positive comparisons'))
        comparisons[candidate+'_vs_'+reference_name] = values
    studies = []
    for folder in sorted((R/'experiments').glob('ac_*structure*window01')):
        if not (folder/'config.json').exists() or not (folder/'independent_audit.json').exists(): continue
        original_cfg = read(folder/'config.json')
        if original_cfg.get('family') != 'convnext_tiny' or not read(folder/'independent_audit.json').get('passed'): continue
        if not list((folder/'evaluations').glob('*/group_audit.json')): continue
        studies.append(folder)
    # Original M is processed first so every later candidate gets its real parent.
    studies.sort(key=lambda p:(p.name!='ac_module_structure_window01',p.name))
    for folder in studies:
        preserve(folder/'config.json'); preserve(folder/'independent_audit.json'); local_names = []
        for group_path in sorted((folder/'evaluations').glob('*/group_audit.json')):
            group = read(group_path); registration = read(group_path.parent/'registration.json')
            assert group['passed'] and group['registration_sha256'] == sha(group_path.parent/'registration.json')
            assert registration['config_sha256'] == sha(folder/'config.json')
            assert registration['source_hashes'] == read(folder/'config.json')['sources']
            preserve(group_path); preserve(group_path.parent/'registration.json')
            for unit in registration['units']:
                assert sha(R/unit['checkpoint']) == unit['checkpoint_sha256']
                audit = read(group_path.parent/f's{unit["seed"]}'/'independent_audit.json')
                assert audit['passed'] and audit['checkpoint_sha256'] == unit['checkpoint_sha256']
                preserve(group_path.parent/f's{unit["seed"]}'/'independent_audit.json')
                assert all(sha(group_path.parent/f's{unit["seed"]}'/p) == h for p,h in audit['prediction_files'].items())
            for mode in ['single','c4']:
                arm = group_path.parent.name; name = folder.name+'/'+arm+'_'+mode
                if folder.name == 'ac_module_structure_window01' and arm == 'c4_moments': name = 'M_'+mode
                cost = group['units'][0]['single_counted_flops']*(4 if mode=='c4' else 1)
                add(name, [group_path.parent/f's{s}'/f'test_{mode}_predictions.csv' for s in [45,46,47]],
                    dict(role='completed CNN network or control; architecture contribution separately assessed',study=folder.name, arm=arm, mode=mode,
                         parameters=group['units'][0]['parameters'],counted_flops=cost,
                         actual_weights=[dict(path=u['checkpoint'],sha256=u['checkpoint_sha256']) for u in registration['units']]))
                local_names.append(name)
        for name in local_names:
            mode = results[name]['mode']; compare(name,'B0_single')
            if mode=='c4': compare(name,'B0_c4')
            if 'M_'+mode in results and name!='M_'+mode: compare(name,'M_'+mode)
        # The first registered arm is the original candidate, not a test-selected winner.
        arms = read(folder/'config.json').get('arms', [])
        if arms:
            for mode in ['single','c4']:
                candidate = folder.name+'/'+arms[0]+'_'+mode
                if candidate in results:
                    for name in local_names:
                        if name != candidate and results[name]['mode']==mode: compare(candidate,name)
        snapshots.append(dict(study=folder.name,methods=len(local_names),complete=True))
        save('status.json',dict(phase='CPU metric reaudit',studies=snapshots,methods=len(results),new_training=0,new_inference=0))
        for name in local_names:
            if name not in ['M_single','M_c4']:
                del cm_cache[name]
                rank_cache.pop(name,None)
    qualified = [dict(comparison=name,metric=k,**v) for name,metrics in comparisons.items() for k,v in metrics.items() if v['conditional_numeric_qualified']]
    save('metric_results.json',dict(passed=True,evaluate_sha256=sha(reference),evaluate_function='exact AST compute_metrics',sklearn=sklearn.__version__,class_names=NAMES,methods=results,comparisons=comparisons,qualified_conditional_numerical_comparisons=qualified,
        primary_source='evaluate.py',patient_equal_accuracy_not_completion_metric=True,no_new_training_or_inference=True,
        limitation='User-authorized posthoc metric scope expansion; unadjusted adaptive model/metric selection; numeric qualification does not automatically confirm 2-3 architectural contributions or novelty',overall_research_complete=False))
    save('input_manifest.json',dict(files=manifest,original_weights_fresh_SHA_verified=True,original_test_indices=ids,patients=41,all_exact_confusion_and_rank_metrics_verified=True))
    save('complete.json',dict(passed=True,methods=len(results),studies=len(studies),result_sha256=sha(S/'metric_results.json'),manifest_sha256=sha(S/'input_manifest.json'),new_training=0,new_inference=0,overall_research_complete=False))


if __name__=='__main__':
    try: main()
    except Exception as error:
        save('failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise

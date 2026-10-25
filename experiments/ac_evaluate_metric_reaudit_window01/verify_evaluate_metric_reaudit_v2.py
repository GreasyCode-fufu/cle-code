"""Independent local NumPy/Fraction audit; no sklearn, torch, or inference."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path, PurePosixPath
from fractions import Fraction
from functools import lru_cache
import csv
import hashlib
import io
import json
import math
import tarfile
import numpy as np

O = Path('outputs/CLE新结构研究/evaluate指标补审')
P = 'experiments/ac_evaluate_metric_reaudit_window01/'
sha = lambda b: hashlib.sha256(b).hexdigest()
control = json.loads(Path('work/CLE_RESEARCH_CONTROL.json').read_text(encoding='utf-8'))
assert control['owner_thread_id'] == os.environ['CODEX_THREAD_ID'] == '01a171be-ebee-7a13-b05e-97fdb6b75587'
receipt = json.loads((O/'export_receipt.json').read_text(encoding='utf-8'))
authority_receipt = json.loads((O/'authority_export_receipt.json').read_text(encoding='utf-8'))
for filename, rec in [('evaluate_metric_reaudit_evidence.tar.gz', receipt), ('evaluate_metric_authority_evidence.tar.gz', authority_receipt)]:
    value = (O/filename).read_bytes()
    assert sha(value) == rec['sha256'] and len(value) == rec['bytes']

with tarfile.open(O/'evaluate_metric_reaudit_evidence.tar.gz') as package, tarfile.open(O/'evaluate_metric_authority_evidence.tar.gz') as supplement:
    def raw(name):
        target = package if name in package.getnames() else supplement
        return target.extractfile(name).read()
    def read(name): return json.loads(raw(name))
    for archive, manifest_name, rec in [(package,'EVIDENCE_MANIFEST.json',receipt),(supplement,'AUTHORITY_MANIFEST.json',authority_receipt)]:
        manifest = json.loads(archive.extractfile(manifest_name).read())
        assert len(manifest['files']) == rec['files']
        assert len(archive.getnames()) == len(set(archive.getnames())) == rec['files']+1
        for entry in manifest['files']:
            value = archive.extractfile(entry['path']).read()
            assert len(value) == entry['bytes'] and sha(value) == entry['sha256']
    assert read('AUTHORITY_MANIFEST.json')['parent_archive_sha256'] == receipt['sha256']
    cfg = read(P+'config.json'); done = read(P+'complete.json'); source_manifest = read(P+'input_manifest.json')
    assert sha(raw(P+'config.json')) == control['auxiliary_audit_config_sha256'] == 'db6d034e5f2a36f0381f423fa9bafa54a0bec8a44063257cc9d41703aa31ec75'
    for name, key in [('audit_evaluate_metrics.py','source_sha256'),('metric_scope_policy.json','policy_sha256'),('PROTOCOL_ZH.md','protocol_sha256'),('register_audit.py','registration_sha256')]:
        assert sha(raw(P+name)) == cfg[key]
    assert sha(raw(P+'evaluate_reference.py')) == cfg['evaluate_sha256'] == 'e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7'
    assert done['passed'] and done['result_sha256'] == sha(raw(P+'metric_results.json')) and done['manifest_sha256'] == sha(raw(P+'input_manifest.json'))
    for name, digest in {**cfg['locked_authority'], **source_manifest['files']}.items(): assert sha(raw(name)) == digest
    results = read(P+'metric_results.json'); methods = results['methods']
    resolution=json.loads((O/'metric_sign_resolution_v2.json').read_text(encoding='utf-8'))
    assert resolution['passed'] and resolution['parent_archive_sha256']==receipt['sha256'] and resolution['original_metric_result_sha256']==sha(raw(P+'metric_results.json'))
    assert len(resolution['changes'])==4 and resolution['affected_qualified_candidates']==0
    for change in resolution['changes']:
        entry=results['comparisons'][change['comparison']][change['metric']]
        assert entry['all_seeds_positive']==change['old_all_seeds_positive'] and not entry['conditional_numeric_qualified']
        entry['all_seeds_positive']=change['corrected_all_seeds_positive']
    assert len(methods) == done['methods'] == 108 and done['studies'] == 11
    assert results['patient_equal_accuracy_not_completion_metric'] and results['no_new_training_or_inference']
    data_cfg = read('experiments/ac_dataset_v1/dataset_config.json')
    ids = data_cfg['indices']['test']; authority = {v['index']:v for v in read('experiments/ac_dataset_v1/dataset_manifest.json')}
    labels = np.array([authority[i]['label'] for i in ids]); groups = [authority[i]['group_id'] for i in ids]
    unique = sorted(set(groups)); group_ids = np.array([unique.index(g) for g in groups])
    assert len(ids) == 576 and len(unique) == 41
    weights = np.random.default_rng(20261024).multinomial(41,np.full(41,1/41),size=20000)
    probabilities = {}; exacts = {}; original_scores = {}; oracle_max_error = 0.; CI_max_error = 0.; CI_checked = 0

    def point_scores(prob):
        pred = prob.argmax(1); cm = np.zeros((3,3),dtype=np.int64)
        np.add.at(cm,(labels,pred),1)
        total = int(cm.sum()); rr = cm.sum(1); cc = cm.sum(0); tp = cm.diagonal()
        def ratio(a,b): return Fraction(int(a),int(b)) if b else Fraction()
        precision = [ratio(tp[c],cc[c]) for c in range(3)]
        recall = [ratio(tp[c],rr[c]) for c in range(3)]
        f1 = [ratio(2*tp[c],rr[c]+cc[c]) for c in range(3)]
        spec = [ratio(total-rr[c]-cc[c]+tp[c],total-rr[c]) for c in range(3)]
        npv = [ratio(total-rr[c]-cc[c]+tp[c],total-cc[c]) for c in range(3)]
        out = dict(overall_accuracy=ratio(tp.sum(),total),macro_precision=sum(precision)/3,
                   macro_recall=sum(recall)/3,macro_sensitivity=sum(recall)/3,balanced_accuracy=sum(recall)/3,
                   macro_specificity=sum(spec)/3,macro_f1=sum(f1)/3,
                   weighted_precision=sum(precision[c]*int(rr[c]) for c in range(3))/total,
                   weighted_recall=sum(recall[c]*int(rr[c]) for c in range(3))/total,
                   weighted_f1=sum(f1[c]*int(rr[c]) for c in range(3))/total)
        chance = sum(int(rr[c])*int(cc[c]) for c in range(3))
        out['kappa'] = ratio(total*tp.sum()-chance,total*total-chance)
        aucs=[]; aps=[]
        for c in range(3):
            for metric, values in [('precision',precision),('recall',recall),('sensitivity',recall),('specificity',spec),('npv',npv),('f1',f1)]:
                out[f'class{c}.{metric}'] = values[c]
            score = prob[:,c]; positive = labels==c; order = np.argsort(score,kind='stable')
            ends = np.r_[np.flatnonzero(np.diff(score[order])!=0)+1,len(labels)]
            twice_rank_sum = 0; start = 0
            for end in ends:
                twice_rank_sum += int(positive[order[start:end]].sum())*(start+1+int(end));start=int(end)
            npos=int(positive.sum()); nneg=len(labels)-npos
            auc=Fraction(twice_rank_sum-npos*(npos+1),2*npos*nneg)
            ap=Fraction()
            for threshold in np.unique(score):
                selected=score>=threshold; positives_at_tie=int((positive & (score==threshold)).sum())
                if positives_at_tie:ap+=Fraction(positives_at_tie,npos)*Fraction(int((positive & selected).sum()),int(selected.sum()))
            out[f'class{c}.auroc']=auc;out[f'class{c}.auc_pr']=ap;aucs.append(auc);aps.append(ap)
        out['auc_roc_macro']=sum(aucs)/3;out['auc_pr_macro']=sum(aps)/3
        denominator=math.sqrt((total*total-int(np.square(rr).sum()))*(total*total-int(np.square(cc).sum())))
        mcc=(total*int(tp.sum())-chance)/denominator if denominator else 0.
        return out,mcc

    for name, entry in methods.items():
        ps=[]; fractions=[]; scores=[]
        assert len(entry['paths'])==3
        for seed,path in zip([45,46,47],entry['paths']):
            rows=list(csv.DictReader(io.StringIO(raw(path).decode())))
            assert len(rows)==576 and [int(v['index']) for v in rows]==ids
            assert all(int(v['label'])==authority[i]['label'] and v['group_id']==authority[i]['group_id'] for i,v in zip(ids,rows))
            prefix='' if 'p0' in rows[0] else 'mean_'
            prob=np.array([[float(v[f'{prefix}p{k}']) for k in range(3)] for v in rows])
            assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1 and np.allclose(prob.sum(1),1,atol=1e-6)
            exact,mcc=point_scores(prob); values={k:float(v) for k,v in exact.items()};values['mcc']=mcc
            assert set(values)==set(entry['metrics'])
            for k,v in values.items():
                error=abs(v-entry['metrics'][k]['seeds'][seed-45]);oracle_max_error=max(oracle_max_error,error);assert error<1e-12,(name,k)
            assert {k:str(v) for k,v in exact.items()}==entry['exact_fractions'][seed-45]
            if 'actual_weights' in entry:
                root=str(PurePosixPath(path).parents[1])+'/'
                registration=read(root+'registration.json');group=read(root+'group_audit.json')
                assert group['passed'] and group['registration_sha256']==sha(raw(root+'registration.json'))
                original_cfg=str(PurePosixPath(root).parents[1])+'/config.json'
                assert registration['config_sha256']==sha(raw(original_cfg))
                assert registration['source_hashes']==read(original_cfg)['sources']
                unit=registration['units'][seed-45];audit=read(str(PurePosixPath(path).parent)+'/independent_audit.json')
                assert audit['passed'] and audit['checkpoint_sha256']==unit['checkpoint_sha256']==entry['actual_weights'][seed-45]['sha256']
                assert unit['checkpoint']==entry['actual_weights'][seed-45]['path'] and audit['prediction_files'][PurePosixPath(path).name]==sha(raw(path))
            ps.append(prob);fractions.append(exact);scores.append(values)
        probabilities[name]=ps;exacts[name]=fractions;original_scores[name]=scores
        for key,v in entry['metrics'].items():assert abs(float(np.mean([s[key] for s in scores]))-v['mean'])<1e-12

    @lru_cache(maxsize=16)
    def cm_distributions(name):
        counts=np.zeros((3,41,9),dtype=np.int64)
        for slot,prob in enumerate(probabilities[name]):np.add.at(counts[slot],(group_ids,labels*3+prob.argmax(1)),1)
        cm=(weights@counts.transpose(1,0,2).reshape(41,27)).reshape(20000,3,3,3)
        tp=cm.diagonal(axis1=2,axis2=3); rr=cm.sum(3);cc=cm.sum(2);n=rr.sum(2)
        def div(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
        precision=div(tp,cc);recall=div(tp,rr);f1=div(2*tp,rr+cc)
        spec=div(n[:,:,None]-rr-cc+tp,n[:,:,None]-rr);npv=div(n[:,:,None]-rr-cc+tp,n[:,:,None]-cc)
        out=dict(overall_accuracy=div(tp.sum(2),n),macro_f1=f1.mean(2),macro_precision=precision.mean(2),
                 macro_recall=recall.mean(2),macro_sensitivity=recall.mean(2),balanced_accuracy=recall.mean(2),
                 macro_specificity=spec.mean(2),weighted_precision=div((precision*rr).sum(2),n),
                 weighted_recall=div((recall*rr).sum(2),n),weighted_f1=div((f1*rr).sum(2),n))
        for c in range(3):
            for key,value in [('precision',precision),('recall',recall),('sensitivity',recall),('f1',f1),('specificity',spec),('npv',npv)]:out[f'class{c}.{key}']=value[:,:,c]
        return {k:v.mean(1) for k,v in out.items()}

    @lru_cache(maxsize=10)
    def rank_distributions(name):
        output={}; auc_seeds=[];ap_seeds=[]
        for c in range(3):
            auc_values=[];ap_values=[];positive=labels==c
            pcount=np.bincount(group_ids[positive],minlength=41);ncount=np.bincount(group_ids[~positive],minlength=41)
            np_draw=weights@pcount;nn_draw=weights@ncount
            for prob in probabilities[name]:
                score=prob[:,c];pair=score[positive,None]-score[None,~positive];pair_matrix=np.zeros((41,41))
                np.add.at(pair_matrix,(np.repeat(group_ids[positive],int((~positive).sum())),np.tile(group_ids[~positive],int(positive.sum()))),((pair>0)+.5*(pair==0)).ravel())
                numerator=((weights@pair_matrix)*weights).sum(1)
                auc_values.append(np.divide(numerator,np_draw*nn_draw,out=np.full(20000,np.nan),where=np_draw*nn_draw!=0))
                levels=np.unique(score)[::-1];level_id=np.searchsorted(-levels,-score)
                increment=np.zeros((len(levels),41),dtype=np.int64);all_increment=np.zeros_like(increment)
                np.add.at(increment,(level_id[positive],group_ids[positive]),1);np.add.at(all_increment,(level_id,group_ids),1)
                positive_cumulative=np.cumsum(increment,axis=0);all_cumulative=np.cumsum(all_increment,axis=0);ap=[]
                for start in range(0,20000,256):
                    w=weights[start:start+256];tp=w@positive_cumulative.T;total=w@all_cumulative.T;delta=w@increment.T
                    prec=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total!=0)
                    ap.extend(np.divide((prec*delta).sum(1),np_draw[start:start+len(w)],out=np.full(len(w),np.nan),where=np_draw[start:start+len(w)]!=0).tolist())
                ap_values.append(np.array(ap))
            output[f'class{c}.auroc']=np.mean(auc_values,axis=0);output[f'class{c}.auc_pr']=np.mean(ap_values,axis=0)
        output['auc_roc_macro']=np.mean([output[f'class{c}.auroc'] for c in range(3)],axis=0)
        output['auc_pr_macro']=np.mean([output[f'class{c}.auc_pr'] for c in range(3)],axis=0)
        return output

    for comparison, entries in results['comparisons'].items():
        left,right=comparison.rsplit('_vs_',1);left_ci=cm_distributions(left);right_ci=cm_distributions(right)
        for key,value in entries.items():
            deltas=[a[key]-b[key] for a,b in zip(original_scores[left],original_scores[right])];mean=float(np.mean(deltas))
            assert max(abs(x-y) for x,y in zip(deltas,value['paired_seed_deltas']))<1e-12 and abs(mean-value['mean_delta'])<1e-12
            if key in exacts[left][0]:
                exact=sum((a[key]-b[key] for a,b in zip(exacts[left],exacts[right])),Fraction())/3
                assert str(exact)==value['exact_mean_fraction']
            if value['conditional_ci95'] is not None:
                if key in left_ci:distribution=left_ci[key]-right_ci[key]
                else:distribution=rank_distributions(left)[key]-rank_distributions(right)[key]
                ci=np.nanpercentile(distribution,[2.5,97.5]);error=float(np.max(np.abs(ci-np.array(value['conditional_ci95']))))
                CI_max_error=max(CI_max_error,error);CI_checked+=1;assert error<1e-12,(comparison,key,error)
                assert int(np.isfinite(distribution).sum())==value['valid_bootstrap_draws']
            eligible=key not in ['kappa','mcc']
            if key in exacts[left][0]:
                positive=all(a[key]>b[key] for a,b in zip(exacts[left],exacts[right]))
            else:
                positive=all(a-b>0 for a,b in zip(methods[left]['metrics'][key]['seeds'],methods[right]['metrics'][key]['seeds']))
            assert value['all_seeds_positive']==positive,(comparison,key)
            assert value['at_least_1p5pp']==bool(eligible and value['mean_delta']>=.015)
            qualified=bool(value['at_least_1p5pp'] and positive and value['conditional_ci95'] is not None and value['conditional_ci95'][0]>0)
            assert qualified==value['conditional_numeric_qualified']
    baseline=read(P+'baseline_weight_bytes_reaudit.json');assert baseline['passed'] and baseline['metric_result_sha256']==sha(raw(P+'metric_results.json'))
    base_reg='experiments/ac_test_feedback_window01/methods_v2/qualification_convnext_tiny_A.json'
    assert baseline['baseline_registration_sha256']==sha(raw(base_reg))
    for unit,original in zip(baseline['units'],read(base_reg)['units']):assert unit['seed']==original['seed'] and unit['checkpoint']==original['checkpoint'] and unit['sha256']==original['checkpoint_sha256']
    for name in ['metric_results.json','input_manifest.json','complete.json','config.json','PROTOCOL_ZH.md','baseline_weight_bytes_reaudit.json']:(O/name).write_bytes(raw(P+name))
    screenshot_policy=json.loads(Path('work/CLE_METRIC_SCOPE_20261025.json').read_text(encoding='utf-8'))
    assert screenshot_policy['version']==4
    allowed=set(screenshot_policy['displayed_completion_proportional_metrics'])
    score_rows=[]
    for comparison,entries in results['comparisons'].items():
        if not comparison.endswith('_vs_B0_single'):continue
        for metric,value in entries.items():
            if value['conditional_numeric_qualified'] and metric in allowed:score_rows.append(dict(comparison=comparison,metric=metric,**value))
    (O/'qualified_numeric_comparisons.json').write_text(json.dumps(score_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

audit=dict(passed=True,main_archive_sha256=receipt['sha256'],authority_archive_sha256=authority_receipt['sha256'],files=receipt['files']+authority_receipt['files'],
           all_file_SHA_verified=True,methods=108,studies=11,all324_prediction_CSVs_576_indices_labels_and_patients_verified=True,
           all_point_metrics_independent_NumPy_Fraction_verified=True,point_max_error=oracle_max_error,
           conditional_CIs_independently_recomputed=CI_checked,CI_max_error=CI_max_error,
           original_group_registration_source_and_actual_weight_SHA_chain_verified=True,server_weight_bytes_rehashed=True,local_weight_bytes_downloaded=False,
           evaluate_reference_sha256=cfg['evaluate_sha256'],patient_equal_accuracy_excluded_from_completion=True,
           nonqualified_exact_zero_sign_metadata_corrected=4,qualified_candidates_changed_by_sign_resolution=0,
           resolution_sidecar_sha256=sha((O/'metric_sign_resolution_v2.json').read_bytes()),screenshot_metric_scope_version=4,
           numeric_qualifications_do_not_automatically_confirm_2to3_architectural_contributions=True,new_training=0,new_inference=0,overall_research_complete=False)
(O/'local_metric_reaudit_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit))

"""Supplementary exact evaluate.py metrics; no neural training or inference."""
from pathlib import Path
from fractions import Fraction
import ast, csv, hashlib, importlib.util, json, os, sys
import numpy as np

F=Path(__file__).resolve().parent; R=F.parents[1]
S=R/'experiments/ac_focal_pairing_dropmom_control_window01'
OLD=R/'experiments/ac_midstage_focal_structure_window01'
EA=R/'experiments/ac_evaluate_metric_reaudit_window01'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    def clean(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:clean(z) for k,z in v.items()}
        if isinstance(v,(list,tuple)):return [clean(z) for z in v]
        return v
    p=F/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
    cfg=read(F/'config.json');assert cfg['sources'][Path(__file__).name]==sha(__file__)
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert read(S/'independent_audit.json')['passed'] and read(S/'bootstrap_complete.json')['passed']
    assert sha(S/'config.json')==cfg['study_config_sha256']
    assert all(sha(R/p)==h for p,h in cfg['locked'].items())
    source=EA/'audit_evaluate_metrics.py'
    spec=importlib.util.spec_from_file_location('frozen_metric_helpers',source);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    reference=EA/'evaluate_reference.py';assert sha(reference)==sha(R/'evaluate.py')==cfg['evaluate_sha256']
    node=next(n for n in ast.parse(reference.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_metrics')
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Name) and n.id in ['open','exec','eval','__import__','os','sys'] for n in ast.walk(node))
    namespace=dict(vars(h));exec(compile(ast.Module(body=[node],type_ignores=[]),str(reference),'exec'),namespace);metric=namespace['compute_metrics']
    data=read(R/'experiments/ac_dataset_v1/dataset_config.json');authority={v['index']:v for v in read(R/'experiments/ac_dataset_v1/dataset_manifest.json')}
    ids=data['indices']['test'];y=np.array([authority[i]['label'] for i in ids]);groups=np.array([authority[i]['group_id'] for i in ids]);assert len(ids)==576 and len(set(groups))==41
    draws=np.random.default_rng(20261024).multinomial(41,np.full(41,1/41),size=20000)
    manifest={};weight_proof={};methods={};prob={};distributions={}
    def preserve(p):
        assert p.is_file();manifest[str(p.relative_to(R))]=sha(p)
    def weight(u):
        p=R/u['checkpoint'];actual=sha(p);assert actual==u['checkpoint_sha256']
        weight_proof[str(p.relative_to(R))]=dict(sha256=actual,bytes=p.stat().st_size,uid=p.stat().st_uid)
    def add(name,paths,meta):
        ps=[];values=[];exact=[];cms=[]
        for path in paths:
            preserve(path);rows=list(csv.DictReader(path.open()));assert [int(r['index']) for r in rows]==ids
            assert all(int(r['label'])==authority[i]['label'] and r['group_id']==authority[i]['group_id'] for i,r in zip(ids,rows))
            prefix='' if 'p0' in rows[0] else 'mean_';p=np.array([[float(r[f'{prefix}p{k}']) for k in range(3)] for r in rows])
            assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and np.allclose(p.sum(1),1,atol=1e-6)
            result=metric(y,p.argmax(1),p,3,h.NAMES);flat=h.flatten(result)
            e={**h.exact_cm(result['confusion_matrix']),**h.exact_rank(p,y)}
            assert all(abs(float(v)-flat[k])<1e-12 for k,v in e.items())
            ps.append(p);values.append(flat);exact.append({k:str(v) for k,v in e.items()});cms.append(result['confusion_matrix'])
        assert len(ps)==3
        methods[name]=dict(**meta,paths=[str(p.relative_to(R)) for p in paths],metrics={k:dict(seeds=[v[k] for v in values],mean=float(np.mean([v[k] for v in values]))) for k in values[0]},exact_fractions=exact,confusion_matrices=cms)
        prob[name]=ps
        distributions[name]={**h.bootstrap_cm(ps,y,groups,draws),**h.bootstrap_rank(ps,y,groups,draws)}
        save('status.json',dict(phase='full screenshot metrics and conditional CI',methods=len(methods),new_training=0,new_inference=0))
    T=R/'experiments/ac_test_feedback_window01'
    for label,method in [('B0_single','qualification_convnext_tiny_A'),('DINO_single','dino_baseline')]:
        reg=T/'methods_v2'/f'{method}.json';preserve(reg);units=read(reg)['units']
        for u in units:weight(u)
        add(label,[T/'predictions'/u['fingerprint']/'test_predictions.csv' for u in units],dict(role='fixedB0' if label.startswith('B0') else 'strong pretrained reference',mode='single'))
    # Same-C4 strong budget evidence is reused, never counted as another architecture.
    for label,root in [('B0_c4',R/'experiments/ac_c4_budget_controls_window01/controls/B0'),('DINO_c4',R/'experiments/ac_c4_budget_controls_window01/controls/DINO'),('Swin_c4',R/'experiments/ac_orientation_reference_window01')]:
        add(label,[root/'runs'/f's{s}'/'test_predictions.csv' for s in [45,46,47]],dict(role='mature four-forward reference',mode='c4'))
    def group(study,arm,label):
        c=read(study/'config.json');folder=study/'evaluations'/arm;g=read(folder/'group_audit.json');reg=read(folder/'registration.json')
        assert g['passed'] and g['registration_sha256']==sha(folder/'registration.json') and reg['config_sha256']==sha(study/'config.json') and reg['source_hashes']==c['sources']
        for p in [study/'config.json',study/'independent_audit.json',folder/'group_audit.json',folder/'registration.json']:preserve(p)
        for p,digest in c['sources'].items():assert sha(R/p)==digest;preserve(R/p)
        for u in reg['units']:
            weight(u);dest=folder/f's{u["seed"]}';audit=read(dest/'independent_audit.json');assert audit['passed'] and audit['checkpoint_sha256']==u['checkpoint_sha256']
            preserve(dest/'independent_audit.json')
            for p,digest in audit['prediction_files'].items():assert sha(dest/p)==digest;preserve(dest/p)
            run=study/'runs'/u['job']
            for n in ['complete.json','validation_complete.json','train_log.csv','validation_predictions.csv']:preserve(run/n)
            assert sha(run/'complete.json')==u['complete_sha256'] and sha(run/'train_log.csv')==u['train_log_sha256']
        for mode in ['single','c4']:
            add(label+'_'+mode,[folder/f's{s}'/f'test_{mode}_predictions.csv' for s in [45,46,47]],dict(role='network or matched structural control',mode=mode,arm=arm,study=str(study.relative_to(R)),parameters=g['units'][0]['parameters'],counted_flops=g['units'][0]['single_counted_flops']*(4 if mode=='c4' else 1),actual_weights=[dict(path=u['checkpoint'],sha256=u['checkpoint_sha256']) for u in reg['units']]))
    group(R/'experiments/ac_module_structure_window01','c4_moments','Mom')
    for arm,label in [('mom_misaligned','Selected'),('mom_focalcontrast','Aligned'),('mom_focalraw','Raw'),('mom_additive','Additive'),('focalcontrast_only','Aligned_noMom')]:group(OLD,arm,label)
    group(S,'misaligned_only','DropMom')
    comparisons={}
    for mode in ['single','c4']:
        selected='Selected_'+mode
        refs=['B0_single','Mom_'+mode,'DropMom_'+mode,'Aligned_'+mode,'Raw_'+mode,'Additive_'+mode,'Aligned_noMom_'+mode,'DINO_single']
        if mode=='c4':refs+=['B0_c4','DINO_c4','Swin_c4']
        for ref in refs:
            a=methods[selected];b=methods[ref];comparison={}
            for key,av in a['metrics'].items():
                bv=b['metrics'][key];ds=[x-z for x,z in zip(av['seeds'],bv['seeds'])]
                if key in a['exact_fractions'][0]:
                    ed=[Fraction(x[key])-Fraction(z[key]) for x,z in zip(a['exact_fractions'],b['exact_fractions'])];delta=sum(ed,Fraction())/3;positive=all(d>0 for d in ed)
                else:ed=None;delta=None;positive=all(d>0 for d in ds)
                mean=float(delta) if delta is not None else av['mean']-bv['mean']
                assert abs(mean-(av['mean']-bv['mean']))<1e-12
                dist=distributions[selected][key]-distributions[ref][key] if key in distributions[selected] else None
                ci=np.nanpercentile(dist,[2.5,97.5]).tolist() if dist is not None else None
                eligible=key in read(S/'metric_scope_policy_v4.json')['displayed_completion_proportional_metrics']
                comparison[key]=dict(mean_delta=mean,paired_seed_deltas=ds,exact_seed_fractions=[str(z) for z in ed] if ed is not None else None,exact_mean_fraction=str(delta) if delta is not None else None,all_seeds_positive=positive,conditional_ci95=ci,valid_draws=int(np.isfinite(dist).sum()) if dist is not None else None,screenshot_proportional_metric=eligible,conditional_numeric_qualified=bool(eligible and delta>=Fraction(3,200) and positive and ci is not None and ci[0]>0))
            comparisons[selected+'_vs_'+ref]=comparison
    for p in [EA/'config.json',EA/'complete.json',EA/'metric_results.json',EA/'local_metric_reaudit_verification.json',EA/'metric_sign_resolution_v2.json',EA/'audit_evaluate_metrics.py',reference,S/'metric_scope_policy_v4.json']:preserve(p)
    save('actual_weight_bytes_reaudit.json',dict(passed=True,weights=weight_proof,server_fresh_bytes_SHA=True,no_weights_downloaded_locally=True))
    save('input_manifest.json',dict(files=manifest,test_indices=ids,patients=41))
    result=dict(passed=True,methods=methods,comparisons=comparisons,evaluate_sha256=cfg['evaluate_sha256'],scope='user screenshot v4',fixed_test=576,paired_seeds=[45,46,47],bootstrap=dict(patients=41,replicates=20000,seed=20261024,not_native_evaluate_image1000=True,model_and_metric_selection_bias_corrected=False),selected_historically_posthoc=True,coefficient_CI='Kappa/MCC point estimates and coefficient deltas reported; not classification percentage-point gate',new_training_in_analysis=0,new_inference_in_analysis=0,overall_research_complete=False)
    save('metric_results.json',result)
    mainkeys=['macro_f1','overall_accuracy','auc_roc_macro','weighted_f1','balanced_accuracy','macro_sensitivity','macro_specificity','macro_precision','auc_pr_macro','kappa','mcc']
    lines=['# 真正去Mom控制与截图全指标补审','','原固定测试576图/41患者；三seed45/46/47，原val最早宏F1最佳权重。Selected是结果后选中的原固定错配控制，不是历史事前选中，也不证明aligned Focal机制。','', '|方案|'+'|'.join(mainkeys)+'|','|---|'+'|'.join(['---:']*len(mainkeys))+'|']
    for name,value in methods.items():lines.append('|'+name+'|'+'|'.join(f'{value["metrics"][k]["mean"]:.7f}' for k in mainkeys)+'|')
    lines+=['','|Selected single每类指标|B0|Mom|Selected|真正DropMom|','|---|---:|---:|---:|---:|']
    for c in range(3):
        for key in ['sensitivity','specificity','precision','f1','auroc']:
            k=f'class{c}.{key}';lines.append('|'+k+'|'+'|'.join(f'{methods[n+"_single"]["metrics"][k]["mean"]:.7f}' for n in ['B0','Mom','Selected','DropMom'])+'|')
    lines+=['','所有逐seed全部指标/混淆矩阵、全部直接Fraction、所有比例指标配对条件区间及成熟控制见metric_results.json，不只展示最佳数值。','', '|Selected single良性Sensitivity比较|差(百分点)|95%条件区间(百分点)|三seed均正|','|---|---:|---|---|']
    for ref in ['B0_single','Mom_single','DropMom_single','Aligned_single','Raw_single','Additive_single','DINO_single']:
        v=comparisons['Selected_single_vs_'+ref]['class0.sensitivity'];lo,hi=v['conditional_ci95'];lines.append(f'|{ref}|{100*v["mean_delta"]:.7f}|[{100*lo:.7f},{100*hi:.7f}]|{v["all_seeds_positive"]}|')
    lines+=['','对B0达1.5pp仅为指定比例指标的条件数值结果；真正累计/去一与成熟参照、整体指标取舍和成本必须结合判断。C4单列四次推理，不算网络组件。Kappa/MCC用系数变化；FPS单列效率。','所有测试均授权自适应内部开发，历史test已查看且AC构建用其标签；模型及指标均结果后选择，41患者20k条件区间未校正选择偏差，不是evaluate.py原逐图1000bootstrap，不称独立/外部验证或临床可靠性证明。旧108方案和所有阴性结果保持。普通阴性不中止整体研究，当前文件不自动宣布整体完成。']
    (F/'METRIC_CONTROL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    save('analysis_complete.json',dict(passed=True,result_sha256=sha(F/'metric_results.json'),input_manifest_sha256=sha(F/'input_manifest.json'),weight_reaudit_sha256=sha(F/'actual_weight_bytes_reaudit.json'),overall_research_complete=False))

if __name__=='__main__':main()

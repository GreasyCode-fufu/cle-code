"""Separate evidence auditor. Recomputes scores/selection/bootstrap from raw artifacts."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]
sys.path[:0]=[str(STUDY),str(ROOT),str(ROOT/'experiments')]
import run_window01 as study


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def state_sha(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def scores(prob,y,groups):
    pred=prob.argmax(1)
    return dict(macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        accuracy=float(accuracy_score(y,pred)),malignant_recall=float(np.mean(pred[y==1]==1)),
        patient_equal_accuracy=float(np.mean([np.mean(pred[groups==g]==y[groups==g]) for g in sorted(set(groups))])),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())


def f1_from_cm(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean(-1)


def compare(a,b,y,groups,draws=20000):
    unique=sorted(set(groups));matrices=[]
    for models in [a,b]:
        each=[]
        for p in models:
            pred=p.argmax(1)
            each.append(np.array([confusion_matrix(y[groups==g],pred[groups==g],labels=[0,1,2]) for g in unique]))
        matrices.append(np.array(each))
    rng=np.random.default_rng(20261024)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=draws)
    values=[]
    for m in matrices:
        cm=np.einsum('dg,sgij->dsij',weights,m)
        values.append(f1_from_cm(cm).mean(1))
    delta=values[0]-values[1]
    seeds=[scores(p,y,groups)['macro_f1']-scores(q,y,groups)['macro_f1'] for p,q in zip(a,b)]
    return dict(seed_deltas=seeds,mean_delta=float(np.mean(seeds)),ci95=np.percentile(delta,[2.5,97.5]).tolist(),
        family_ci=np.percentile(delta,[.5,99.5]).tolist(),groups=len(unique),draws=draws,
        all_seed_positive=all(d>0 for d in seeds),strong_dev_signal=all(d>0 for d in seeds) and np.mean(seeds)>=.005 and np.percentile(delta,.5)>0,
        interpretation='Conditional patient resampling of selected fitted models; repeated validation/AC selection/training uncertainty not removed.')


def main():
    cfg=study.verify()
    for p,s in cfg['locked_files'].items():
        assert sha(ROOT/p)==s
    data=read(ROOT/'experiments/ac_dataset_v1/dataset_config.json')
    ac=read(ROOT/'experiments/ac_dataset_v1/dataset_manifest.json')
    records=read(STUDY/'manifest.json')
    assert len(records)==3595 and len(ac)==3080
    sets={s:{r['group_id'] for r in ac if r['split']==s} for s in ['train','val','test']}
    assert [len(sets[s]) for s in ['train','val','test']]==[182,39,41]
    assert all(not sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')])
    val=data['indices']['val']; y=np.array([records[i]['label'] for i in val]);groups=np.array([records[i]['group_id'] for i in val])
    allpred=collections.defaultdict(list);units=[]
    jobs=read(STUDY/'jobs.json')
    work=[(j['arm'],j['seed'],STUDY/'runs'/j['id'],j) for j in jobs]
    work += [('baseline',s,STUDY/'references'/f's{s}',None) for s in [45,46,47]]
    for arm,seed,folder,job in work:
        comp=read(folder/'complete.json')
        if job:
            assert study.signature(job)==comp['signature'] and job['train']==data['indices']['train'] and job['select']==val
        else:
            ref=next(r for r in cfg['references'] if r['seed']==seed)
            assert comp['signature']==ref['source_signature']
            assert all(sha(folder/n)==v for n,v in ref['metadata_sha256'].items())
        ck=ROOT/comp['checkpoint'];assert sha(ck)==comp['checkpoint_sha256']
        log=list(csv.DictReader((folder/'train_log.csv').open()))
        assert len(log)==60 and [int(r['epoch']) for r in log]==list(range(60))
        best=max(range(60),key=lambda i:float(log[i]['val_macro_f1']))
        assert best==comp['best_epoch'] and float(log[best]['val_macro_f1'])==comp['best_select_f1']
        if job:
            index_sha=hashlib.sha256(json.dumps(sorted(job['train'])).encode()).hexdigest()
            assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 and r['train_index_set_sha256']==index_sha for r in log)
            assert all(np.isfinite(float(r[k])) for r in log for k in ['train_loss','train_acc','val_macro_f1','loader_wait_seconds','compute_seconds'])
        rows=list(csv.DictReader((folder/'validation_predictions.csv').open()))
        assert [int(r['index']) for r in rows]==val
        assert [int(r['label']) for r in rows]==y.tolist()
        assert [r['group_id'] for r in rows]==groups.tolist()
        saved=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])
        assert np.isfinite(saved).all() and np.allclose(saved.sum(1),1,atol=1e-6) and (saved>=0).all()
        model=study.model_for(seed,arm)
        state=torch.load(ck,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
        model.load_state_dict(state,strict=True)
        count=sum(p.numel() for p in model.parameters());assert count==comp['parameters']
        before=state_sha(model)
        p,actual_y,idx=study.core.infer(model,study.loader(val,False,seed))
        assert idx==val and np.array_equal(y,actual_y) and np.max(np.abs(p-saved))<1e-6
        assert state_sha(model)==before
        m=scores(p,y,groups);assert abs(m['macro_f1']-comp['best_select_f1'])<1e-12
        diagnostic=None
        if getattr(model,'readout',None) is not None:
            model.readout.enabled=False
            off,_,_=study.core.infer(model,study.loader(val,False,seed))
            diagnostic=dict(off_metrics=scores(off,y,groups),prediction_changed_fraction=float(np.mean(off.argmax(1)!=p.argmax(1))),
                            max_probability_difference=float(np.max(np.abs(off-p))))
            study.core.save_csv(folder/'diagnostic_off_predictions.csv',study.core.prediction_rows(off,y,val))
        allpred[arm].append(p)
        units.append(dict(arm=arm,seed=seed,checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=sha(ck),
            prediction_sha256=sha(folder/'validation_predictions.csv'),parameters=count,best_epoch=best,
            finite_weights=True,strict_reload=True,max_probability_error=float(np.max(np.abs(p-saved))),state_unchanged=True,
            metrics=m,seconds=comp['seconds'],diagnostic=diagnostic))
        del state,model;torch.cuda.empty_cache()
    comparisons={}
    for reference in ['baseline','no_neighbor','no_depth','final_only','official4']:
        comparisons['relational_vs_'+reference]=compare(allpred['relational'],allpred[reference],y,groups)
    summary={}
    for arm in ['baseline',*study.ARMS]:
        items=[u for u in units if u['arm']==arm]
        summary[arm]={k:float(np.mean([u['metrics'][k] for u in items])) for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']}
        summary[arm]['seed_macro_f1']=[u['metrics']['macro_f1'] for u in items]
        summary[arm]['parameters']=items[0]['parameters']
    errors=[]
    for pos,i in enumerate(val):
        r=records[i]
        errors.append(dict(index=i,group_id=r['group_id'],label=r['label'],sample_id=r['sample_id'],
            **{a+'_error_count':sum(int(p[pos].argmax()!=y[pos]) for p in allpred[a]) for a in allpred}))
    study.core.save_csv(STUDY/'validation_error_evidence.csv',errors)
    study.save(STUDY/'paired_comparisons.json',comparisons)
    study.save(STUDY/'round1_results.json',dict(summary=summary,units=units))
    own_paths=[STUDY,study.OWN_MODEL,study.core.WEIGHTS]
    wrong=[]
    for root in own_paths:
        paths=[root,*root.rglob('*')] if root.is_dir() else [root]
        wrong.extend(str(p) for p in paths if p.stat().st_uid!=1000)
    assert not wrong
    study.save(STUDY/'independent_audit.json',dict(passed=True,timestamp=study.core.timestamp(),n_models=len(units),
        new_trainings=18,reused_baselines=3,checks=['frozen data and patient isolation','60epoch earliest-best selection',
        'runtime actual training index sets','checkpoint/source SHA','all finite tensors/strict reload','all427 independent probabilities',
        'independent sklearn metrics and patient equal scores','independent confusion-matrix bootstrap','readout-off diagnostics',
        'parameter budgets','login ownership'],test_images_loaded=0,source_hashes=study.source_hashes()))
    lines=['# CLE 新窗口第一轮开发结果','', '本轮完成并独立审计；整个研究目标仍在进行。未运行新的 test 核验。','',
           '| 方案 | seed45/46/47 宏F1 | 均值 | 恶性召回 | 患者等权准确率 | 参数 |','|---|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('|'+arm+'|'+ '/'.join(f'{x:.6f}' for x in s['seed_macro_f1'])+f'|{s["macro_f1"]:.6f}|{s["malignant_recall"]:.6f}|{s["patient_equal_accuracy"]:.6f}|{s["parameters"]}|')
    lines += ['', '主要候选相对baseline：'+json.dumps(comparisons['relational_vs_baseline'],ensure_ascii=False),'',
        '未通过预注册开发门槛不能宣称机制成立。即使开发门槛通过，也不能代替最终test三seed平均严格>1.5个百分点、稳定且患者区间下界为正。',
        'AC定义使用过test标签且历史test已查看；本轮仍仅使用train/val，不称全新独立外部验证。',
        '完整阴性、源签名、checkpoint SHA、逐图预测、off诊断与所有seed见同目录JSON/CSV。下一轮必须依据开发错误证据提出不同机制并预注册；不自动循环查看test。']
    (STUDY/'ROUND1_REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()

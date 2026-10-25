"""Independent data, checkpoint, probability and patient-metric reconstruction."""
from pathlib import Path
import csv,gc,json,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_module_structure as m
import numpy as np
import torch
sys.path.insert(0,str(m.T))
from audit_test_feedback_v2 import state_sha,compute_metrics,conditional_bootstrap

def readpred(path):
    rows=list(csv.DictReader(path.open()));prefix='' if all(f'p{k}' in rows[0] for k in range(3)) else 'mean_';assert all(f'{prefix}p{k}' in rows[0] for k in range(3));return rows,np.array([[float(r[f'{prefix}p{k}']) for k in range(3)] for r in rows])

def raw_infer(model,indices,rotations):
    from PIL import Image
    from torchvision.transforms import functional as tf
    allowed=set(m.READ(S/'config.json')['indices']['val']+m.READ(S/'config.json')['indices']['test']);assert set(indices)<=allowed
    array=np.load(m.R/'.cache_data_study/resized_uint8.npy',mmap_mode='r');assert array.shape==(3595,256,256,3)
    perview=[[] for _ in rotations];model.eval()
    with torch.inference_mode():
        for begin in range(0,len(indices),32):
            batch=[tf.normalize(tf.to_tensor(tf.center_crop(Image.fromarray(array[i]),[224,224])),[.485,.456,.406],[.229,.224,.225]) for i in indices[begin:begin+32]]
            x=torch.stack(batch).cuda()
            for slot,k in enumerate(rotations):
                z=torch.rot90(x,k,(-2,-1)).contiguous()
                with torch.autocast('cuda',dtype=torch.float16):logits=model(z)
                perview[slot].append(logits.float().softmax(1).cpu().numpy())
    return np.stack([np.concatenate(v) for v in perview])

def savepred(path,p,indices,records):
    rows=[dict(index=i,sample_id=records[i]['sample_id'],group_id=records[i]['group_id'],label=records[i]['label'],pred=int(z.argmax()),**{f'p{k}':float(z[k]) for k in range(3)}) for i,z in zip(indices,p)]
    if path.exists():
        old,persisted=readpred(path)
        assert [int(r['index']) for r in old]==indices and np.max(np.abs(p-persisted))<1e-6
        assert all(r['sample_id']==records[i]['sample_id'] and r['group_id']==records[i]['group_id'] and int(r['label'])==records[i]['label'] and int(r['pred'])==int(z.argmax()) for r,i,z in zip(old,indices,p))
    else:m.core.save_csv(path,rows)

def audit_group(arm,jobs):
    cfg=m.verify();folder=S/'evaluations'/arm;folder.mkdir(parents=True,exist_ok=True);units=[]
    trainhash=m.hashlib.sha256(json.dumps(sorted(cfg['indices']['train'])).encode()).hexdigest()
    for job in jobs:
        root=S/'runs'/job['id'];c=m.READ(root/'complete.json');log=list(csv.DictReader((root/'train_log.csv').open()))
        assert m.done(job) and c['signature']==m.signature(job) and len(log)==c['epochs_completed']==60
        assert [int(r['epoch']) for r in log]==list(range(60))
        best=max(range(60),key=lambda n:float(log[n]['val_macro_f1']));assert best==c['best_epoch']
        assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 and r['train_index_set_sha256']==trainhash for r in log)
        units.append(dict(seed=job['seed'],job=job['id'],checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],signature=c['signature'],best_epoch=best,complete_sha256=m.sha(root/'complete.json'),train_log_sha256=m.sha(root/'train_log.csv')))
    assert [u['seed'] for u in units]==m.SEEDS
    registration=dict(audit_revision=2,audit_revision_config_sha256=m.sha(S/'postprocessing_revision_v2.json'),arm=arm,units=units,source_hashes=m.sources(),config_sha256=m.sha(S/'config.json'),test_views=[0,1,2,3],selection='original0degree val earliest maximum; C4 test never selects epoch')
    rp=folder/'registration.json'
    if rp.exists():assert m.READ(rp)==registration
    else:m.SAVE(rp,registration);m.commit('research: freeze all3 val-selected '+arm+' checkpoints before adaptive test feedback')
    records={r['index']:r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([records[i]['label'] for i in ids]);groups=np.array([records[i]['group_id'] for i in ids]);audits=[]
    for job,unit in zip(jobs,units):
        dest=folder/f's{job["seed"]}';dest.mkdir(exist_ok=True);done=dest/'independent_audit.json'
        if done.exists():
            a=m.READ(done);assert a['passed'] and a['checkpoint_sha256']==unit['checkpoint_sha256']
            assert all(m.sha(dest/p)==h for p,h in a['prediction_files'].items());audits.append(a);continue
        start=time.perf_counter();model=m.model_for(job['seed'],arm);state=torch.load(m.R/unit['checkpoint'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state;before=state_sha(model)
        val=raw_infer(model,cfg['indices']['val'],[0])[0];saved,expected=readpred(S/'runs'/job['id']/'validation_predictions.csv');verror=float(np.max(np.abs(val-expected)))
        assert [int(r['index']) for r in saved]==cfg['indices']['val'] and verror<1e-6
        vy=np.array([records[i]['label'] for i in cfg['indices']['val']]);vg=np.array([records[i]['group_id'] for i in cfg['indices']['val']]);vm=compute_metrics(val,vy,vg)
        assert abs(vm['macro_f1']-float(m.READ(S/'runs'/job['id']/'complete.json')['best_select_f1']))<1e-12
        views=raw_infer(model,ids,[0,1,2,3]);assert views.shape==(4,576,3) and np.isfinite(views).all() and np.allclose(views.sum(-1),1,atol=1e-6)
        # Entirely separate original TestData route, every image/angle, for a loader oracle.
        from run_test_feedback_v2 import TestData
        dl=torch.utils.data.DataLoader(TestData('A'),batch_size=32,shuffle=False,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(job['seed']))
        repro=[[] for _ in range(4)];seen=[];yy=[];model.eval()
        with torch.inference_mode():
            for x,labels,_,index in dl:
                seen.extend(index.tolist());yy.extend(labels.tolist());x=x.cuda(non_blocking=True)
                for k in range(4):
                    with torch.autocast('cuda',dtype=torch.float16):logits=model(torch.rot90(x,k,(-2,-1)).contiguous())
                    repro[k].append(logits.float().softmax(1).cpu().numpy())
        reproduction=np.stack([np.concatenate(v) for v in repro]);terror=float(np.max(np.abs(reproduction-views)))
        assert seen==ids and yy==y.tolist() and terror<1e-6 and state_sha(model)==before and m.sha(m.R/unit['checkpoint'])==unit['checkpoint_sha256']
        np.save(dest/'test_views.npy',views);single=views[0];average=views.mean(0)
        savepred(dest/'test_single_predictions.csv',single,ids,records);savepred(dest/'test_c4_predictions.csv',average,ids,records)
        budget=next(b for b in m.READ(S/'preflight.json')['budgets'] if b['arm']==arm);assert sum(p.numel() for p in model.parameters())==budget['parameters']
        a=dict(passed=True,seed=job['seed'],arm=arm,checkpoint_sha256=unit['checkpoint_sha256'],prediction_files={p:m.sha(dest/p) for p in ['test_views.npy','test_single_predictions.csv','test_c4_predictions.csv']},val_max_probability_error=verror,test_max_probability_error=terror,test_images=576,test_patients=len(set(groups)),state_unchanged=True,state_sha256=before,
            val_metrics=vm,single_metrics=compute_metrics(single,y,groups),c4_metrics=compute_metrics(average,y,groups),parameters=budget['parameters'],single_counted_flops=budget['counted_flops'],c4_counted_flops=4*budget['counted_flops'],seconds=time.perf_counter()-start,independent_generalization_confirmation=False)
        m.SAVE(done,a);audits.append(a);del model,views,val;gc.collect();torch.cuda.empty_cache()
    m.SAVE(folder/'group_audit.json',dict(passed=True,arm=arm,registration_sha256=m.sha(rp),units=audits));m.commit('results: retain independently reconstructed single/C4 test feedback for '+arm+' all3 seeds')

def summarize():
    cfg=m.verify();records={r['index']:r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([records[i]['label'] for i in ids]);g=np.array([records[i]['group_id'] for i in ids]);ps={};summary={}
    def add(name,files,role,cost=None):
        rows=[];ps[name]=[]
        for seed,path in zip(m.SEEDS,files):
            rr,p=readpred(path);assert [int(r['index']) for r in rr]==ids
            assert [int(r['label']) for r in rr]==y.tolist() and [r['group_id'] for r in rr]==g.tolist()
            ps[name].append(p);rows.append(dict(seed=seed,**compute_metrics(p,y,g)))
        summary[name]=dict(role=role,seeds=rows,seed_macro_f1=[r['macro_f1'] for r in rows],**{k:float(np.mean([r[k] for r in rows])) for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']},counted_flops=cost)
    for name in ['qualification_convnext_tiny_A','dino_baseline']:
        reg=m.READ(m.T/'methods_v2'/f'{name}.json');add(name,[m.T/'predictions'/u['fingerprint']/'test_predictions.csv' for u in reg['units']],'fixed B0' if 'tiny' in name else 'strong historical DINO')
    for label,root in [('B0_C4',m.R/'experiments/ac_c4_budget_controls_window01/controls/B0'),('DINO_C4',m.R/'experiments/ac_c4_budget_controls_window01/controls/DINO'),('Swin_C4',m.R/'experiments/ac_orientation_reference_window01')]:
        add(label,[root/'runs'/f's{s}'/'test_predictions.csv' for s in m.SEEDS],'mature C4 budget/control reference')
    budgets={b['arm']:b for b in m.READ(S/'preflight.json')['budgets']}
    for arm in m.ARMS:
        if not (S/'evaluations'/arm/'group_audit.json').exists():continue
        for mode in ['single','c4']:
            add(arm+'_'+mode,[S/'evaluations'/arm/f's{s}'/f'test_{mode}_predictions.csv' for s in m.SEEDS],'learned mechanism/reference',budgets[arm]['counted_flops']*(4 if mode=='c4' else 1))
    comparisons={}
    for name,item in summary.items():
        if item['role']!='learned mechanism/reference':continue
        refs=['qualification_convnext_tiny_A','B0_C4','dino_baseline','DINO_C4','Swin_C4']
        if name.startswith('c4_moments_') and 'c4_aug_'+name.rsplit('_',1)[1] in ps:refs.append('c4_aug_'+name.rsplit('_',1)[1])
        for ref in refs:
            delta=np.array(item['seed_macro_f1'])-np.array(summary[ref]['seed_macro_f1']);ci=conditional_bootstrap(ps[name],ps[ref],y,g)
            comparisons[name+'_vs_'+ref]=dict(paired_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),conditional_patient_bootstrap=ci,numerical_gate=bool(delta.mean()>.015 and (delta>0).all() and ci['ci95'][0]>0),independent_generalization_confirmation=False)
    result=dict(updated_at=m.core.timestamp(),baseline=cfg['baseline'],summary=summary,comparisons=comparisons,budgets=budgets,test_role=cfg['test_role'],selection_bias_corrected=False,overall_research_complete=False);m.SAVE(S/'round_results.json',result)
    lines=['# 共享模块与方向机制正式实验（持续研究）','','固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。','', '|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in summary.items():lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in r['seed_macro_f1']+[r[k] for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']])+'|')
    lines+=['','|比较|平均差（百分点）|患者条件95%区间（百分点）|','|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    lines+=['','逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。']
    (S/'MODULE_ROUND_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def evaluate_completed_groups():
    jobs=m.READ(S/'jobs.json');changed=False
    for arm in m.ARMS:
        if (S/'evaluations'/arm/'group_audit.json').exists():continue
        group=[j for j in jobs if j['arm']==arm]
        if all((S/'runs'/j['id']/'validation_complete.json').exists() for j in group):audit_group(arm,group);changed=True
    if changed:summarize();m.commit('analysis: record every paired module mechanism, budget control and negative result')

def final_audit():
    cfg=m.verify();assert all((S/'evaluations'/arm/'group_audit.json').exists() for arm in m.ARMS)
    assert m.sha(m.R/'.cache_data_study/resized_uint8.npy')==m.READ(m.R/'.cache_data_study/meta.json')['cache_sha256']
    rec=m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json');gs={split:{r['group_id'] for r in rec if r['split']==split} for split in ['train','val','test']}
    assert [len(gs[s]) for s in ['train','val','test']]==[182,39,41] and all(not gs[a]&gs[b] for a,b in [('train','val'),('train','test'),('val','test')])
    for arm in m.ARMS:
        folder=S/'evaluations'/arm;a=m.READ(folder/'group_audit.json');assert a['passed'] and a['registration_sha256']==m.sha(folder/'registration.json')
        for unit in m.READ(folder/'registration.json')['units']:
            dest=folder/f's{unit["seed"]}';audit=m.READ(dest/'independent_audit.json');assert m.sha(m.R/unit['checkpoint'])==audit['checkpoint_sha256']
            assert all(m.sha(dest/p)==h for p,h in audit['prediction_files'].items())
    assert all(p.stat().st_uid==1000 for root in [S,m.M,m.R/'weights/ac_module_structure_window01'] for p in [root,*root.rglob('*')])
    m.SAVE(S/'independent_audit.json',dict(passed=True,models=18,seeds=m.SEEDS,train=2077,val=427,test=576,patient_isolation=True,finite_parameters=True,earliest_val_best=True,actual_train_indices_exact=True,full_val_test_probability_reconstruction=True,single_and_c4_budgets=True,login_ownership=True,independent_generalization_confirmation=False,overall_research_complete=False));summarize()

if __name__=='__main__':evaluate_completed_groups()

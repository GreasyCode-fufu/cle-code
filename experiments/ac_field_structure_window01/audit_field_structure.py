"""Independent full val/test reconstruction before adaptive feedback is reported."""
from pathlib import Path
import argparse,csv,gc,json,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S))
import run_field_structure as f
sys.path.insert(0,str(f.T))
from audit_test_feedback_v2 import compute_metrics,conditional_bootstrap,state_sha
import numpy as np
import torch

def predictions(folder):
    rows=list(csv.DictReader((folder/'test_predictions.csv').open()))
    return rows,np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])

def independent_infer(model,indices,recipe):
    # Independent center transform and support construction; no field training loader.
    from PIL import Image
    from torchvision.transforms import functional as F
    resize,crop=(256,224) if recipe=='A' else (448,384)
    original=np.load(R/'.cache_data_study/resized_uint8.npy',mmap_mode='r')
    qmap=f.READ(f.Q/'cache448_meta.json');qm={i:p for p,i in enumerate(qmap['indices'])}
    qarr=np.load(f.Q/'cache448/images.npy',mmap_mode='r')
    tm=f.READ(f.T/'cache_audit.json');tp={i:p for p,i in enumerate(tm['indices'])};ta=np.load(f.T/'cache/test448.npy',mmap_mode='r')
    yy=(np.arange(resize)+.5)*2/resize-1;xx=(np.arange(resize)+.5)*2/resize-1
    mask=((yy[:,None]**2+xx[None,:]**2)<=1).astype(np.uint8)*255
    support=F.to_tensor(F.center_crop(Image.fromarray(mask),[crop,crop]));ps=[];model.eval()
    with torch.inference_mode():
        for begin in range(0,len(indices),32):
            batch=[]
            for i in indices[begin:begin+32]:
                array=original[i] if recipe=='A' else (qarr[qm[i]] if i in qm else ta[tp[i]])
                image=F.center_crop(Image.fromarray(array),[crop,crop]);rgb=F.normalize(F.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
                batch.append(torch.cat((rgb,support),0))
            x=torch.stack(batch).cuda()
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
            ps.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(ps)

def audit_group(kind,jobs):
    cfg=f.verify();folder=S/'evaluations'/kind;folder.mkdir(parents=True,exist_ok=True)
    registration=dict(kind=kind,family=cfg['family'],recipe=cfg['recipe'],config_sha256=f.sha(S/'config.json'),sources=f.sources(),
        protocol='all3 complete60 epochs, earliest val-best checkpoint locked before test',units=[])
    trainhash=f.hashlib.sha256(json.dumps(sorted(cfg['indices']['train'])).encode()).hexdigest()
    for job in jobs:
        path=S/'runs'/job['id'];c=f.READ(path/'complete.json');log=list(csv.DictReader((path/'train_log.csv').open()))
        assert f.done(job) and c['signature']==f.signature(job) and len(log)==c['epochs_completed']==60
        assert [int(r['epoch']) for r in log]==list(range(60))
        best=max(range(60),key=lambda n:float(log[n]['val_macro_f1']));assert c['best_epoch']==best
        assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 and r['train_index_set_sha256']==trainhash for r in log)
        registration['units'].append(dict(seed=job['seed'],job_id=job['id'],signature=c['signature'],best_epoch=best,
            checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],complete_sha256=f.sha(path/'complete.json'),train_log_sha256=f.sha(path/'train_log.csv')))
    assert [u['seed'] for u in registration['units']]==f.SEEDS
    rp=folder/'registration.json'
    if rp.exists():assert f.READ(rp)==registration
    else:f.SAVE(rp,registration);f.commit('research: lock full paired '+kind+' val-selected checkpoints before test feedback')
    rows={r['index']:r for r in f.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')};units=[]
    for job,unit in zip(jobs,registration['units']):
        dest=folder/f's{job["seed"]}';dest.mkdir(exist_ok=True);done=dest/'independent_audit.json'
        if done.exists():
            audit=f.READ(done);assert audit['passed'] and audit['checkpoint_sha256']==unit['checkpoint_sha256']
            assert f.sha(dest/'test_predictions.csv')==audit['prediction_sha256'];units.append(audit);continue
        start=time.perf_counter();model=f.model_for(job['seed'],kind);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state
        before=state_sha(model);v=independent_infer(model,cfg['indices']['val'],cfg['recipe'])
        saved=list(csv.DictReader((S/'runs'/job['id']/'validation_predictions.csv').open()))
        expected=np.array([[float(r[f'p{k}']) for k in range(3)] for r in saved]);valerr=float(np.max(np.abs(v-expected)))
        assert [int(r['index']) for r in saved]==cfg['indices']['val'] and valerr<1e-6
        vy=np.array([rows[i]['label'] for i in cfg['indices']['val']]);vg=np.array([rows[i]['group_id'] for i in cfg['indices']['val']]);vm=compute_metrics(v,vy,vg)
        assert abs(vm['macro_f1']-float(f.READ(S/'runs'/job['id']/'complete.json')['best_select_f1']))<1e-12
        p=independent_infer(model,cfg['indices']['test'],cfg['recipe']);ty=np.array([rows[i]['label'] for i in cfg['indices']['test']]);tg=np.array([rows[i]['group_id'] for i in cfg['indices']['test']])
        assert p.shape==(576,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
        target=dest/'test_predictions.csv'
        if target.exists():
            old,oldp=predictions(dest);assert [int(r['index']) for r in old]==cfg['indices']['test'] and np.max(np.abs(oldp-p))<1e-6
        else:
            prediction_rows=[]
            for i,z in zip(cfg['indices']['test'],p):
                row=rows[i];prediction_rows.append(dict(index=i,sample_id=row['sample_id'],group_id=row['group_id'],label=row['label'],pred=int(z.argmax()),**{f'p{k}':float(z[k]) for k in range(3)}))
            f.core.save_csv(target,prediction_rows)
        # Re-infer through separately constructed paired loader and compare every probability.
        from run_test_feedback_v2 import TestData
        from paired_field_data import PairedFieldDataset
        shape=f.q.RECIPES[cfg['recipe']];data=PairedFieldDataset(TestData(cfg['recipe']),resize=shape['resize'],crop=shape['crop'],training=False)
        dl=torch.utils.data.DataLoader(data,batch_size=32,shuffle=False,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(job['seed']))
        repro,yy,ii=f.core.infer(model,dl);testerr=float(np.max(np.abs(repro-p)))
        assert ii==cfg['indices']['test'] and np.array_equal(yy,ty) and testerr<1e-6
        assert state_sha(model)==before and f.sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
        gates={name:float(parameter.detach().cpu()) for name,parameter in model.named_parameters() if name.endswith('.alpha')}
        audit=dict(passed=True,seed=job['seed'],kind=kind,checkpoint_sha256=unit['checkpoint_sha256'],prediction_sha256=f.sha(target),
            test_images=576,test_patients=len(set(tg)),val_images=427,val_max_probability_error=valerr,test_max_probability_error=testerr,
            state_unchanged=True,state_sha256=before,metrics=compute_metrics(p,ty,tg),val_metrics=vm,gate_values=gates,
            parameters=sum(x.numel() for x in model.parameters()),seconds=time.perf_counter()-start,
            independent_generalization_confirmed=False,scope='Independent computational reconstruction only; test adaptive selection remains.')
        f.SAVE(done,audit);units.append(audit);del model,v,p,repro;gc.collect();torch.cuda.empty_cache()
    f.SAVE(folder/'group_audit.json',dict(passed=True,kind=kind,registration_sha256=f.sha(rp),units=units,paired_seeds=f.SEEDS))
    f.commit('results: independently audit all3 seeds and retain complete '+kind+' test feedback')

def summarize():
    cfg=f.verify();ref=f.READ(S/'working_reference.json');prior=f.READ(S/'evidence/test_feedback_snapshot.json')
    ac={r['index']:r for r in f.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([ac[i]['label'] for i in ids]);g=np.array([ac[i]['group_id'] for i in ids])
    methods={};ps={};summary={}
    for reference in [ref['method'],'qualification_'+cfg['primary_baseline']]:
        units=prior['summary'][reference]['seeds'];ps[reference]=[]
        for u in units:_,p=predictions(f.T/'predictions'/u['fingerprint']);ps[reference].append(p)
        summary[reference]=dict(role='matched mature W0' if reference==ref['method'] else 'fixed primary B0',**prior['summary'][reference])
    for kind in f.ARMS:
        p=S/'evaluations'/kind/'group_audit.json'
        if not p.exists():continue
        audit=f.READ(p);assert audit['passed'];units=audit['units'];ps[kind]=[predictions(p.parent/f's{s}')[1] for s in f.SEEDS]
        summary[kind]=dict(role='mechanism/control',seed_macro_f1=[u['metrics']['macro_f1'] for u in units],
            **{metric:float(np.mean([u['metrics'][metric] for u in units])) for metric in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']},units=units)
    comparisons={};references=[ref['method'],'qualification_'+cfg['primary_baseline']]
    for name,m in summary.items():
        if name.startswith('qualification_'):continue
        for reference in set(references+['fov_residual']):
            if reference not in ps or reference==name:continue
            d=np.array(m['seed_macro_f1'])-np.array(summary[reference]['seed_macro_f1']);ci=conditional_bootstrap(ps[name],ps[reference],y,g)
            comparisons[name+'_vs_'+reference]=dict(paired_seed_deltas=d.tolist(),mean_delta=float(d.mean()),conditional_patient_bootstrap=ci,
                numerical_gate=bool(d.mean()>.015 and (d>0).all() and ci['ci95'][0]>0),independent_confirmation=False)
    budget={b['arm']:b for b in f.READ(S/'preflight.json')['budgets']}
    result=dict(updated_at=f.core.timestamp(),primary_B0=cfg['primary_baseline'],working_W0=ref['arm'],summary=summary,comparisons=comparisons,budgets=budget,
        test_role='adaptive development feedback',selection_bias_corrected=False,independent_generalization_confirmed=False,overall_research_complete=False)
    f.SAVE(S/'round_results.json',result)
    lines=['# 几何视场支持正式结构研究','',f'固定B0：{cfg["primary_baseline"]}；同主干预算参照W0：{ref["arm"]}。W0按完整资格方案test均值选择，明确为适应性开发，不能称独立确认。','',
        '|方法|seed45 F1|seed46 F1|seed47 F1|平均F1|accuracy|恶性召回|患者等权accuracy|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in summary.items():lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in m['seed_macro_f1']+[m[k] for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']])+'|')
    lines+=['','|配对比较|平均差（百分点）|患者条件95%区间（百分点）|','|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    lines+=['','逐seed混淆矩阵、门值、checkpoint SHA、逐图预测与预算见round_results.json及evaluations。历史test已被查看，A–C构建用过test标签；等级是模型一致性代理。现test参与方法选择，条件bootstrap不消除选择偏差。成熟算子组合不自动构成SCI创新；普通阴性后继续有证据的机制调整。']
    (S/'FIELD_ROUND_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result

def evaluate_completed_groups():
    jobs=f.READ(S/'jobs.json');changed=False
    for kind in f.ARMS:
        if (S/'evaluations'/kind/'group_audit.json').exists():continue
        group=[j for j in jobs if j['arm']==kind]
        if all((S/'runs'/j['id']/'validation_complete.json').exists() for j in group):audit_group(kind,group);changed=True
    if changed:summarize();f.commit('analysis: preserve every paired mechanism/control test result and uncertainty')

def final_audit():
    cfg=f.verify();result=summarize();assert all(k in result['summary'] for k in f.ARMS)
    for kind in f.ARMS:
        folder=S/'evaluations'/kind;a=f.READ(folder/'group_audit.json');reg=f.READ(folder/'registration.json')
        assert a['passed'] and a['registration_sha256']==f.sha(folder/'registration.json')
        for u in reg['units']:
            d=folder/f's{u["seed"]}';c=f.READ(d/'independent_audit.json');assert c['passed'] and f.sha(R/u['checkpoint'])==c['checkpoint_sha256']
            assert f.sha(d/'test_predictions.csv')==c['prediction_sha256']
    groups={split:{r['group_id'] for r in f.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json') if r['split']==split} for split in ['train','val','test']}
    assert [len(groups[s]) for s in ['train','val','test']]==[182,39,41] and all(not groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    weights=R/'weights/ac_field_structure_window01';assert all(p.stat().st_uid==1000 for root in [S,weights] for p in [root,*root.rglob('*')])
    f.SAVE(S/'independent_audit.json',dict(passed=True,models=18,seeds=f.SEEDS,train=2077,val=427,test=576,test_patients=41,
        earliest_val_best=True,actual_training_indices_exact=True,finite_parameters=True,source_and_weight_hashes_verified=True,
        probability_reconstruction=True,patient_isolation=True,login_ownership=True,overall_research_complete=False,independent_generalization_confirmed=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');a=parser.parse_args()
    if a.final:final_audit()
    else:evaluate_completed_groups()

"""Fixed C4 probability-mean reference; no training, adaptive test use disclosed."""
from pathlib import Path
import argparse,csv,fcntl,gc,hashlib,json,os,subprocess,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1]
Q=R/'experiments/ac_qualification_20261024_window01';F=R/'experiments/ac_field_structure_window01';T=R/'experiments/ac_test_feedback_window01'
sys.path[:0]=[str(Q),str(T),str(R/'experiments'),str(R)]
import run_qualification as q
import run_test_feedback_v2 as feedback
core=q.core;READ=core.read_json;SAVE=core.save_json
import numpy as np
import torch
from audit_test_feedback_v2 import compute_metrics,state_sha

def sha(p):return core.digest(p)
def sources():return {str(p.relative_to(R)):sha(p) for p in [S/'run_orientation_reference.py',S/'bootstrap_orientation_reference.py',S/'ORIENTATION_REFERENCE_PROTOCOL_ZH.md',Q/'run_qualification.py',Q/'qualification_model.py',T/'run_test_feedback_v2.py',T/'audit_test_feedback_v2.py']}
def commit(message):
    # Keep this pipeline's staging out of the live structure scheduler's completion/audit windows.
    path=str(S.relative_to(R))
    while True:
        staged=subprocess.check_output([core.GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();quiet=all(x.startswith(path+'/') for x in staged)
        manager=READ(F/'queue_manager_identity.json')['pid'];proc=Path('/proc')/str(manager)/'cmdline';live=proc.exists() and b'manage_field_queue_v2.py' in proc.read_bytes()
        if live:
            status=READ(F/'queue_status_v2.json');quiet &= bool(status.get('running'))
            for item in status.get('running',[]):
                p=Path('/proc')/str(item['pid'])/'cmdline';log=F/'runs'/item['job']/'train_log.csv';meta=F/'runs'/item['job']/'run_meta.json'
                n=len(list(csv.DictReader(log.open()))) if log.exists() else 0
                quiet &= p.exists() and item['job'].encode() in p.read_bytes() and n<55 and meta.exists() and time.time()-meta.stat().st_mtime>10
            for arm in ['fov_residual','field_pool','reference','no_field','no_local_norm','no_pool']:
                ready=all((F/'runs'/f'field_{arm}_s{seed}'/'validation_complete.json').exists() for seed in [45,46,47]);audit=F/'evaluations'/arm/'group_audit.json'
                if ready and (not audit.exists() or time.time()-audit.stat().st_mtime<10):quiet=False
        if quiet:break
        print('waiting for clean own-pipeline Git window',flush=True);time.sleep(5)
    subprocess.run([core.GIT,'-C',str(R),'add','--',path],check=True)
    if subprocess.run([core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([core.GIT,'-C',str(R),'commit','-m',message],check=True)

def verify():
    q.verify();feedback.verify();cfg=READ(S/'config.json');assert cfg['sources']==sources()
    for p,h in cfg['locked'].items():assert sha(R/p)==h,p
    return cfg

def prepare():
    if (S/'config.json').exists():return verify()
    assert os.getuid()==1000 and S.stat().st_uid==1000
    q.verify();feedback.verify()
    w=READ(F/'working_reference.json');assert w['arm']=='swin_t_B'
    (S/'reference.json').write_bytes((F/'working_reference.json').read_bytes())
    for u in w['registration']['units']:assert sha(R/u['checkpoint'])==u['checkpoint_sha256']
    ds=READ(R/'experiments/ac_dataset_v1/dataset_config.json');ac=READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')
    rows=[r for r in ac if r['split'] in ['val','test']];SAVE(S/'manifest.json',rows)
    assert len(rows)==1003 and len({r['group_id'] for r in rows if r['split']=='val'})==39 and len({r['group_id'] for r in rows if r['split']=='test'})==41
    locks=[S/'reference.json',S/'manifest.json',F/'evidence/test_feedback_snapshot.json',Q/'config.json',Q/'cache448_meta.json',T/'config_v2.json',T/'cache_audit.json',R/'experiments/ac_dataset_v1/dataset_config.json',R/'experiments/ac_dataset_v1/dataset_manifest.json']
    SAVE(S/'config.json',dict(created_at=core.timestamp(),sources=sources(),locked={str(p.relative_to(R)):sha(p) for p in locks},
        units=w['registration']['units'],family=w['family'],recipe=w['recipe'],arm=w['arm'],seeds=[45,46,47],
        indices={split:ds['indices'][split] for split in ['val','test']},angles=[0,90,180,270],aggregation='arithmetic mean of softmax probabilities; exactly equal fixed weights',
        checkpoint_selection='reuse original earliest val0-best full60-epoch checkpoint; no new checkpoint selection',batch_size=32,amp='float16',workers=2,
        training_images=0,network_updates=0,unique_val_images=427,unique_test_images=576,primary_baseline=READ(F/'config.json')['primary_baseline'],
        inference_compute_multiplier=4,novel_network_contribution=False,test_role='user-authorized adaptive development feedback',independent_confirmation=False))
    commit('research: register fixed C4 mature inference reference and all3 val-selected weight hashes before val-test evaluation')
    return verify()

def bootstrap(a,b,y,groups):
    unique=sorted(set(groups));count=len(unique);assert (len(y),count) in [(427,39),(576,41)]
    gi=np.array([unique.index(group) for group in groups]);rng=np.random.default_rng(20261024);weights=rng.multinomial(count,np.full(count,1/count),size=20000);values=[]
    assert np.all(weights.sum(1)==count)
    for collection in [a,b]:
        matrices=np.zeros((3,count,3,3),dtype=np.int64)
        for seed,p in enumerate(collection):np.add.at(matrices[seed],(gi,y,p.argmax(1)),1)
        cm=np.einsum('dg,sgij->dsij',weights,matrices);tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
        values.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2)))
    d=values[0]-values[1]
    return dict(ci95=np.percentile(d,[2.5,97.5]).tolist(),draws=20000,patients=count,images=len(y),paired_seeds=3,
        draw_group_count_verified=True,scope='Conditional interval only; repeated val/test feedback and AC cohort selection remain.')

def loader(split,seed):
    cfg=READ(S/'config.json')
    data=q.GuardedData(cfg['indices'][split],False,cfg['recipe']) if split=='val' else feedback.TestData(cfg['recipe'])
    return torch.utils.data.DataLoader(data,batch_size=32,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=False,generator=torch.Generator().manual_seed(seed))

def cached(split,unit):
    p=(R/unit['complete']).parent/'validation_predictions.csv' if split=='val' else T/'predictions'/unit['fingerprint']/'test_predictions.csv'
    if split=='test':
        proof=READ(p.parent/'independent_audit.json');assert proof['passed'] and proof['prediction_sha256']==sha(p) and proof['checkpoint_sha256']==unit['checkpoint_sha256']
    else:
        proof=READ(p.parent/'validation_complete.json');assert proof['prediction_sha256']==sha(p) and proof['checkpoint_sha256']==unit['checkpoint_sha256']
    rows=list(csv.DictReader(p.open()));return rows,np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])

def evaluate():
    cfg=verify();assert READ(S/'preflight.json')['passed'];records={r['index']:r for r in READ(S/'manifest.json')}
    for unit in cfg['units']:
        seed=unit['seed'];dest=S/'runs'/f's{seed}';proof=dest/'complete.json'
        if proof.exists():
            saved=READ(proof);assert saved['checkpoint_sha256']==unit['checkpoint_sha256'] and all(sha(dest/name)==h for name,h in saved['files'].items());continue
        dest.mkdir(parents=True,exist_ok=True)
        while q.engine.snapshot()['gpu_free_mib']<8500:time.sleep(5)
        assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];model=q.model_for(seed,cfg['arm']);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state;model.eval();before=state_sha(model)
        stats={};start=time.perf_counter();files={}
        for split in ['val','test']:
            reference,target=cached(split,unit);indices=cfg['indices'][split];assert [int(r['index']) for r in reference]==indices
            actual=[];views=[[] for _ in range(4)];y=[];wait=compute=0.;iterator=iter(loader(split,seed));torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                while True:
                    tick=time.perf_counter()
                    try:x,labels,_,ii=next(iterator)
                    except StopIteration:break
                    wait+=time.perf_counter()-tick;tick=time.perf_counter();x=x.cuda(non_blocking=True)
                    for k in range(4):
                        xx=x if k==0 else torch.rot90(x,k,(-2,-1)).contiguous()
                        with torch.autocast('cuda',dtype=torch.float16):logits=model(xx)
                        views[k].append(logits.float().softmax(1).cpu().numpy())
                    actual.extend(ii.tolist());y.extend(labels.tolist());torch.cuda.synchronize();compute+=time.perf_counter()-tick
                    q.engine.append_resource(S/'resources.csv',dict(**q.engine.snapshot(),seed=seed,split=split,images_done=len(actual)))
            view=np.stack([np.concatenate(v) for v in views]);mean=view.astype(float).mean(0);y=np.array(y);groups=np.array([records[i]['group_id'] for i in indices])
            assert actual==indices and y.tolist()==[records[i]['label'] for i in indices] and len(set(groups))==(39 if split=='val' else 41)
            error=float(np.abs(view[0]-target).max());assert error<1e-6,'Original0-degree factory/loader reference must reproduce'
            assert np.isfinite(view).all() and np.allclose(view.sum(-1),1,atol=1e-6)
            np.save(dest/(split+'_views.npy'),view);preds=[]
            for j,i in enumerate(indices):
                row=records[i];preds.append(dict(index=i,sample_id=row['sample_id'],group_id=row['group_id'],label=row['label'],pred=int(mean[j].argmax()),
                    **{f'mean_p{c}':float(mean[j,c]) for c in range(3)},**{f'angle{angle}_p{c}':float(view[k,j,c]) for k,angle in enumerate(cfg['angles']) for c in range(3)}))
            core.save_csv(dest/(split+'_predictions.csv'),preds)
            metrics={str(angle):compute_metrics(view[k],y,groups) for k,angle in enumerate(cfg['angles'])};metrics['mean_C4']=compute_metrics(mean,y,groups)
            stats[split]=dict(metrics=metrics,original_probability_max_error=error,
                mean_prediction_disagreement_with_0=float(np.mean([np.mean(view[k].argmax(1)!=view[0].argmax(1)) for k in [1,2,3]])),
                any_rotation_disagreement=float(np.mean(np.any(view.argmax(-1)!=view[0].argmax(1)[None],axis=0))),
                loader_wait_seconds=wait,compute_seconds=compute,unique_images=len(indices),image_views=4*len(indices),patients=len(set(groups)),
                peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            for name in [split+'_views.npy',split+'_predictions.csv']:files[name]=sha(dest/name)
        assert state_sha(model)==before
        SAVE(proof,dict(seed=seed,checkpoint_sha256=unit['checkpoint_sha256'],config_sha256=sha(S/'config.json'),source_hashes=sources(),stats=stats,files=files,
            state_sha256=before,state_unchanged=True,network_updates=0,training_images=0,seconds=time.perf_counter()-start,inference_compute_multiplier=4,novel_network_contribution=False))
        del model,view,mean;gc.collect();torch.cuda.empty_cache();commit('results: preserve all angles and fixed C4 reference for seed'+str(seed))

def preflight():
    cfg=verify();assert READ(S/'reference.json')['arm']==cfg['arm']
    x=torch.arange(3*8*8).reshape(1,3,8,8);assert torch.equal(torch.rot90(x,4,(-2,-1)),x)
    try:q.GuardedData(cfg['indices']['test'][:1],False,cfg['recipe'])
    except AssertionError:pass
    else:raise AssertionError('val loader accepted test')
    for split,expected in [('val',(427,39)),('test',(576,41))]:
        rows=[r for r in READ(S/'manifest.json') if r['split']==split];assert (len(rows),len({r['group_id'] for r in rows}))==expected
    SAVE(S/'preflight.json',dict(passed=True,rotation4_identity=True,cohort_counts_verified=True,val_guard_rejects_test=True,equal_probability_weights_frozen=True,network_updates=0))
    commit('test: validate fixed C4 transform identity, data counts and split boundary before inference')

def analyze():
    cfg=verify();records={r['index']:r for r in READ(S/'manifest.json')};units=[];summary={};comparisons={}
    baseline=READ(F/'evidence/test_feedback_snapshot.json')['summary']['qualification_'+cfg['primary_baseline']]
    for unit in cfg['units']:
        folder=S/'runs'/f's{unit["seed"]}';c=READ(folder/'complete.json');assert c['state_unchanged'] and c['checkpoint_sha256']==unit['checkpoint_sha256']
        assert all(sha(folder/name)==h for name,h in c['files'].items());units.append(c)
    for split in ['val','test']:
        indices=cfg['indices'][split];y=np.array([records[i]['label'] for i in indices]);groups=np.array([records[i]['group_id'] for i in indices]);views=[]
        for unit in cfg['units']:
            folder=S/'runs'/f's{unit["seed"]}';view=np.load(folder/(split+'_views.npy'));assert view.shape==(4,len(indices),3);views.append(view)
            rows=list(csv.DictReader((folder/(split+'_predictions.csv')).open()));assert [int(r['index']) for r in rows]==indices and [int(r['label']) for r in rows]==y.tolist()
            assert [r['group_id'] for r in rows]==groups.tolist() and [int(r['pred']) for r in rows]==view.astype(float).mean(0).argmax(1).tolist()
            assert [r['sample_id'] for r in rows]==[records[i]['sample_id'] for i in indices]
            assert np.max(np.abs(np.array([[float(r[f'mean_p{j}']) for j in range(3)] for r in rows])-view.astype(float).mean(0)))<1e-10
            for k,angle in enumerate(cfg['angles']):assert np.max(np.abs(np.array([[float(r[f'angle{angle}_p{j}']) for j in range(3)] for r in rows])-view[k]))<1e-10
        mean=[v.astype(float).mean(0) for v in views];plain=[v[0] for v in views];fm=[compute_metrics(p,y,groups) for p in mean];pm=[compute_metrics(p,y,groups) for p in plain]
        summary[split]=dict(C4_seed_macro_f1=[m['macro_f1'] for m in fm],plain_seed_macro_f1=[m['macro_f1'] for m in pm],
            **{key:float(np.mean([m[key] for m in fm])) for key in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']})
        delta=np.array([m['macro_f1'] for m in fm])-np.array([m['macro_f1'] for m in pm]);comparisons[split+'_C4_vs_plain']=dict(paired_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),conditional_patient_bootstrap=bootstrap(mean,plain,y,groups))
        if split=='test':
            prior=READ(F/'evidence/test_feedback_snapshot.json')['summary']['qualification_'+cfg['primary_baseline']];base=[]
            for u in prior['seeds']:
                rows=list(csv.DictReader((T/'predictions'/u['fingerprint']/'test_predictions.csv').open()));assert [int(r['index']) for r in rows]==indices;base.append(np.array([[float(r[f'p{j}']) for j in range(3)] for r in rows]))
            delta=np.array([m['macro_f1'] for m in fm])-np.array(prior['seed_macro_f1']);ci=bootstrap(mean,base,y,groups)
            comparisons['test_C4_vs_frozen_B0']=dict(paired_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),conditional_patient_bootstrap=ci,numerical_gate=bool(delta.mean()>.015 and (delta>0).all() and ci['ci95'][0]>0),independent_confirmation=False)
    budget=next(b for b in READ(Q/'preflight.json')['budgets'] if b['arm']==cfg['arm'])
    result=dict(created_at=core.timestamp(),summary=summary,comparisons=comparisons,units=units,underlying_parameters=budget['parameters'],counted_flops=4*budget['counted_flops'],
        source_hashes=sources(),config_sha256=sha(S/'config.json'),network_updates=0,training_images=0,method='fixed C4 mean probabilities',novel_network_contribution=False,test_role='adaptive development feedback',independent_confirmation=False,overall_research_complete=False)
    SAVE(S/'orientation_results.json',result);SAVE(S/'independent_computational_audit.json',dict(passed=True,models=3,seeds=[45,46,47],val_unique_images=427,val_patients=39,test_unique_images=576,test_patients=41,
        all_original_probabilities_reproduced=True,all_views_finite=True,all_states_unchanged=True,source_weight_prediction_hashes=True,mean_and_metrics_reconstructed=True,bootstrap_group_count_verified=True,network_updates=0,independent_confirmation=False))
    lines=['# 固定四方向平均推理：成熟参照与方向敏感性','',
        '所有角度、等权概率平均、三seed旧val最佳权重和val/test清单均在推理前登记。本方法无新训练，属于成熟增强推理参照，不算网络创新；方向探测结果用于后续结构机制。test参与开发，条件区间不能称独立确认。','',
        '|split|C4 seed45 F1|seed46|seed47|C4均值|0度均值|','|---|---:|---:|---:|---:|---:|']
    for split,m in summary.items():lines.append('|'+split+'|'+'|'.join(f'{100*v:.4f}%' for v in m['C4_seed_macro_f1']+[m['macro_f1'],float(np.mean(m['plain_seed_macro_f1']))])+'|')
    lines+=['',f'四视角推理成本为原W0的4倍：{4*budget["counted_flops"]/1e9:.5f} GFLOPs（原模型参数不增加）。唯一样本数仍为val427/39患者、test576/41患者，不能把视图数当新图像或新患者。全部角度分数、逐图预测、方向分歧、指标和不确定性见JSON。','',
        '|比较|平均差（百分点）|患者条件95%区间|','|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    lines+=['','不能用本结果宣称提出了新结构。若设计低成本方向稳健模块，必须与原主干、同预算通用适配器、成熟旋转增强/平均推理和重训消融比较；只有经验证的有效组件才能保留后继续叠加。历史test与A–C构建标签依赖照旧披露。']
    (S/'ORIENTATION_REFERENCE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');commit('audit: reconstruct every C4 angle, fixed aggregate and paired baseline uncertainty without new training')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','evaluate','analyze']);a=parser.parse_args()
    lock=(S/'method.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[a.command]()

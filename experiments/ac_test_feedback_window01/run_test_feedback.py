"""User-authorized adaptive test feedback; isolated from all frozen training code."""
from pathlib import Path
import argparse, csv, fcntl, gc, hashlib, json, os, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
S1=ROOT/'experiments/ac_structure_20261024_window01'
Q=ROOT/'experiments/ac_qualification_20261024_window01'
COHORT=ROOT/'experiments/ac_dataset_v1'
GIT='/home/login/miniconda3/bin/git'
sys.path[:0]=[str(S1),str(Q),str(ROOT/'experiments'),str(ROOT)]
import numpy as np
import torch
import run_window01 as first
import run_qualification as qualification
core=first.core
SEEDS=[45,46,47]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return core.digest(p)
def save(p,v):core.save_json(Path(p),v)
def signature(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def sources():return {n:sha(HERE/n) for n in ['run_test_feedback.py','audit_test_feedback.py','bootstrap_test_feedback.py','TEST_FEEDBACK_PROTOCOL.md']}

def safe_commit(message):
    # Qualification's existing commit helper requires an otherwise empty index.
    # Wait for a quiet training window instead of leaving unrelated paths staged.
    while True:
        staged=subprocess.check_output([GIT,'-C',str(ROOT),'diff','--cached','--name-only'],text=True).splitlines()
        status=read(Q/'status.json') if (Q/'status.json').exists() else {}
        quiet=True
        for item in status.get('running',[]):
            p=Path('/proc')/str(item['pid'])/'cmdline'
            try:live=b'run_qualification.py' in p.read_bytes()
            except (FileNotFoundError,PermissionError,ProcessLookupError):live=False
            if live:
                log=Q/'runs'/item['job']/'train_log.csv'
                epochs=len(list(csv.DictReader(log.open()))) if log.exists() else 0
                if epochs>54:quiet=False
        if not staged and quiet:break
        print('waiting for clean Git/qualification commit window',flush=True);time.sleep(10)
    subprocess.run([GIT,'-C',str(ROOT),'add','--',str(HERE.relative_to(ROOT))],check=True)
    if subprocess.run([GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
        subprocess.run([GIT,'-C',str(ROOT),'commit','-m',message],check=True)

def prepare():
    assert os.getuid()==1000 and HERE.stat().st_uid==1000
    if (HERE/'config.json').exists():return verify()
    first.verify();qualification.verify()
    cfg=read(COHORT/'dataset_config.json');ac=read(COHORT/'dataset_manifest.json')
    assert sha(COHORT/'dataset_manifest.json')==cfg['manifest_sha256']
    groups={s:{r['group_id'] for r in ac if r['split']==s} for s in ['train','val','test']}
    for s,n,g in [('train',2077,182),('val',427,39),('test',576,41)]:
        rows=[r for r in ac if r['split']==s]
        assert len(rows)==n and len(groups[s])==g and set(cfg['indices'][s])=={r['index'] for r in rows}
        assert all(r['grade'] in ('A','B','C') for r in rows)
    assert all(not groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    path=cache/'test448.npy';part=cache/'test448.partial.npy'
    rows={r['index']:r for r in ac if r['split']=='test'};indices=cfg['indices']['test']
    old=np.load(ROOT/'.cache_data_study/resized_uint8.npy',mmap_mode='r')
    assert old.shape==(3595,256,256,3)
    out=np.lib.format.open_memmap(part,mode='w+',dtype=np.uint8,shape=(576,448,448,3))
    def load(i):
        r=rows[i];p=ROOT/r['path'];assert sha(p)==r['sha256']
        with Image.open(p) as im:
            im=im.convert('RGB')
            assert np.array_equal(np.asarray(im.resize((256,256),Image.Resampling.BILINEAR)),old[i])
            return np.asarray(im.resize((448,448),Image.Resampling.BILINEAR))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pos,image in enumerate(pool.map(load,indices)):out[pos]=image
    out.flush();del out;os.replace(part,path)
    save(HERE/'cache_audit.json',dict(passed=True,indices=indices,shape=[576,448,448,3],sha256=sha(path),
        original_cache_rows=3595,actual_A_test_access=len(indices),actual_B_test_access=576,
        raw_image_sha_verified=576,original_resize256_pixel_exact=576,patient_intersections_empty=True,
        patient_counts={s:len(g) for s,g in groups.items()},train_or_val_images_in_new_cache=0))
    locked=[COHORT/'dataset_config.json',COHORT/'dataset_manifest.json',COHORT/'independent_dataset_audit.json',
        S1/'config.json',S1/'independent_audit.json',Q/'config.json',HERE/'cache_audit.json']
    save(HERE/'config.json',dict(created_at=core.timestamp(),sources=sources(),
        locked={str(p.relative_to(ROOT)):sha(p) for p in locked},indices=cfg['indices']['test'],seeds=SEEDS,
        protocol='train -> earliest val-best checkpoint -> test -> next-method feedback',
        test_role='adaptive development feedback; not independent confirmation',
        baseline_rule='preserve qualification precommitted validation selection; never weaken B0 using test',
        batch_size=32,amp='float16',workers=2,reserve_gpu_mib=6144,
        bootstrap=dict(replicates=20000,seed=20261024,unit='same41 patients resampled jointly across all paired seeds'),
        threshold=.015,independent_generalization_confirmed=False))
    safe_commit('research: freeze user-authorized per-method test-feedback inference and independent audit')
    return verify()

def verify():
    cfg=read(HERE/'config.json');assert cfg['sources']==sources(),'new test source drift'
    for p,h in cfg['locked'].items():assert sha(ROOT/p)==h,p
    return cfg

class TestData:
    def __init__(self,recipe,indices=None):
        from torchvision import transforms as T
        cfg=read(HERE/'config.json');self.indices=cfg['indices'] if indices is None else indices
        assert self.indices==cfg['indices'],'test inference requires exactly fixed576 indices'
        self.rows={r['index']:r for r in read(COHORT/'dataset_manifest.json') if r['split']=='test'}
        assert set(self.indices)==set(self.rows) and len(self.indices)==576
        self.recipe=recipe;assert recipe in ['A','B']
        self.array=np.load(ROOT/'.cache_data_study/resized_uint8.npy' if recipe=='A' else HERE/'cache/test448.npy',mmap_mode='r')
        self.positions={i:p for p,i in enumerate(self.indices)}
        self.transform=T.Compose([T.CenterCrop(224 if recipe=='A' else 384),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return 576
    def __getitem__(self,pos):
        from PIL import Image
        i=self.indices[pos];j=i if self.recipe=='A' else self.positions[i]
        return self.transform(Image.fromarray(self.array[j])),self.rows[i]['label'],1.,i

def loader(recipe,seed):
    return torch.utils.data.DataLoader(TestData(recipe),batch_size=32,shuffle=False,num_workers=2,pin_memory=True,
        generator=torch.Generator().manual_seed(seed),persistent_workers=False)

def validate_training(folder,job,factory,reference=None):
    comp=read(folder/'complete.json');log=list(csv.DictReader((folder/'train_log.csv').open()))
    assert len(log)==comp['epochs_completed']==60 and [int(r['epoch']) for r in log]==list(range(60))
    best=max(range(60),key=lambda k:float(log[k]['val_macro_f1']))
    assert best==comp['best_epoch'] and abs(float(log[best]['val_macro_f1'])-comp['best_select_f1'])<1e-12
    assert comp['train_count']==2077 and comp['select_count']==427
    if reference:
        assert comp['signature']==reference['source_signature']
        assert all(sha(folder/n)==h for n,h in reference['metadata_sha256'].items())
    else:
        expected=qualification.signature(job) if factory=='qualification' else signature_job_first(job)
        assert comp['signature']==expected
    assert sha(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
    return dict(seed=job['seed'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],
        training_signature=comp['signature'],complete=str((folder/'complete.json').relative_to(ROOT)),complete_sha256=sha(folder/'complete.json'),
        train_log_sha256=sha(folder/'train_log.csv'),best_epoch=best,val_macro_f1=comp['best_select_f1'],parameters=comp['parameters'])

def signature_job_first(job):
    return hashlib.sha256((json.dumps(job,sort_keys=True)+sha(S1/'config.json')).encode()).hexdigest()

def available():
    groups=[];fcfg=read(S1/'config.json');qcfg=read(Q/'config.json')
    refs=[validate_training(S1/'references'/f's{r["seed"]}',dict(seed=r['seed']),'first',r) for r in fcfg['references']]
    groups.append(dict(method='dino_baseline',factory='first',arm='baseline',recipe='A',canonical='dino_cls_mean',units=refs))
    for arm in first.ARMS:
        jobs=[j for j in read(S1/'jobs.json') if j['arm']==arm]
        if all((S1/'runs'/j['id']/'complete.json').exists() for j in jobs):
            units=[validate_training(S1/'runs'/j['id'],j,'first') for j in jobs]
            groups.append(dict(method='round1_'+arm,factory='first',arm=arm,recipe='A',
                canonical='mature_convnext_base' if arm=='convnext' else 'dino_'+arm,units=units))
    for family in qualification.FAMILIES:
        for recipe in ['A','B']:
            arm=family+'_'+recipe
            if arm=='convnext_base_A':
                units=[validate_training(Q/'references'/f'convnext_base_A_s{r["seed"]}',r,'qualification',r) for r in qcfg['references']]
            else:
                jobs=[j for j in read(Q/'jobs.json') if j['arm']==arm]
                if not all((Q/'runs'/j['id']/'complete.json').exists() for j in jobs):continue
                units=[validate_training(Q/'runs'/j['id'],j,'qualification') for j in jobs]
            groups.append(dict(method='qualification_'+arm,factory='qualification',arm=arm,recipe=recipe,canonical='mature_'+family,units=units))
    for group in groups:
        group['units']=sorted(group['units'],key=lambda u:u['seed']);assert [u['seed'] for u in group['units']]==SEEDS
        group['cohort_manifest_sha256']=read(COHORT/'dataset_config.json')['manifest_sha256']
        group['evaluation_config_sha256']=sha(HERE/'config.json')
        for u in group['units']:
            u['fingerprint']=signature(dict(checkpoint=u['checkpoint_sha256'],canonical=group['canonical'],recipe=group['recipe'],
                cohort=group['cohort_manifest_sha256'],evaluation=group['evaluation_config_sha256']))
    return groups

def model_for(group,seed):
    if group['factory']=='first':return first.model_for(seed,group['arm'])
    return qualification.model_for(seed,group['arm'])

def resources():
    import psutil
    line=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return dict(timestamp=core.timestamp(),gpu_util=float(line[0]),gpu_used_mib=float(line[1]),gpu_free_mib=float(line[2]),
        cpu_percent=psutil.cpu_percent(),ram_percent=psutil.virtual_memory().percent)

def run_unit(group,u):
    from audit_test_feedback import audit_unit,compute_metrics,state_sha
    dest=HERE/'predictions'/u['fingerprint'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'independent_audit.json').exists():
        a=read(dest/'independent_audit.json');c=read(dest/'complete.json')
        assert a['passed'] and a['fingerprint']==u['fingerprint'] and a['checkpoint_sha256']==u['checkpoint_sha256']
        assert sha(dest/'test_predictions.csv')==a['prediction_sha256']==c['prediction_sha256'];return
    while resources()['gpu_free_mib']<8500:time.sleep(10)
    start=time.perf_counter();before_resource=resources();model=model_for(group,u['seed'])
    state=torch.load(ROOT/u['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    model.load_state_dict(state,strict=True);model.eval();del state
    assert sum(p.numel() for p in model.parameters())==u['parameters']
    before=state_sha(model);torch.cuda.reset_peak_memory_stats()
    reused=False;old=None
    if group['method']=='dino_baseline':
        old=ROOT/'experiments/ac_final_evaluation/runs'/f'final_round3_acbackbone_dino_full_s{u["seed"]}'
        proof=read(old/'test_complete.json')
        assert proof['checkpoint_sha256']==u['checkpoint_sha256'] and proof['prediction_sha256']==sha(old/'test_predictions.csv')
        assert read(ROOT/'experiments/ac_final_evaluation/independent_audit.json')['passed']
        shutil.copyfile(old/'test_predictions.csv',dest/'test_predictions.csv');reused=True
    if not (dest/'test_predictions.csv').exists():
        prob,y,ix=core.infer(model,loader(group['recipe'],u['seed']))
        ac={r['index']:r for r in read(COHORT/'dataset_manifest.json') if r['split']=='test'}
        assert ix==read(HERE/'config.json')['indices'] and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
        with (dest/'test_predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['index','sample_id','group_id','label','pred','p0','p1','p2']);writer.writeheader()
            for i,label,p in zip(ix,y,prob):
                r=ac[i];assert int(label)==r['label']
                writer.writerow(dict(index=i,sample_id=r['sample_id'],group_id=r['group_id'],label=int(label),pred=int(p.argmax()),
                    **{f'p{k}':float(p[k]) for k in range(3)}))
    assert before==state_sha(model)
    rows=list(csv.DictReader((dest/'test_predictions.csv').open()));p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])
    y=np.array([int(r['label']) for r in rows]);g=np.array([r['group_id'] for r in rows])
    save(dest/'complete.json',dict(fingerprint=u['fingerprint'],unit=u,method=group['method'],recipe=group['recipe'],
        checkpoint_sha256=u['checkpoint_sha256'],prediction_sha256=sha(dest/'test_predictions.csv'),
        metrics=compute_metrics(p,y,g),state_sha256=before,state_unchanged=True,parameters=u['parameters'],
        reused_historical_inference=reused,reused_from=str(old.relative_to(ROOT)) if old else None,
        primary_inference_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        resources_before=before_resource,resources_after=resources(),evaluation_config_sha256=sha(HERE/'config.json')))
    audit_unit(group,u,dest,model)
    del model;gc.collect();torch.cuda.empty_cache()

def summarize(groups):
    from audit_test_feedback import conditional_bootstrap
    units={};summary={};probabilities={}
    for group in groups:
        if not all((HERE/'predictions'/u['fingerprint']/'independent_audit.json').exists() for u in group['units']):continue
        each=[];probs=[]
        for u in group['units']:
            d=HERE/'predictions'/u['fingerprint'];c=read(d/'complete.json');a=read(d/'independent_audit.json');assert a['passed']
            each.append(dict(seed=u['seed'],fingerprint=u['fingerprint'],metrics=c['metrics'],best_epoch=u['best_epoch'],checkpoint=u['checkpoint'],checkpoint_sha256=u['checkpoint_sha256']))
            rows=list(csv.DictReader((d/'test_predictions.csv').open()));probs.append(np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]))
            units[u['fingerprint']]=dict(complete_sha256=sha(d/'complete.json'),audit_sha256=sha(d/'independent_audit.json'))
        summary[group['method']]=dict(recipe=group['recipe'],canonical=group['canonical'],seeds=each,
            seed_macro_f1=[u['metrics']['macro_f1'] for u in each],
            **{k:float(np.mean([u['metrics'][k] for u in each])) for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']})
        probabilities[group['method']]=probs
    baseline=None
    selected=Q/'selected_baseline.json'
    if selected.exists() and (Q/'independent_audit.json').exists() and read(Q/'independent_audit.json')['passed']:
        b=read(selected);assert b['frozen'];baseline='qualification_'+b['selected']['arm']
        proof=dict(method=baseline,selected_baseline_sha256=sha(selected),qualification_audit_sha256=sha(Q/'independent_audit.json'),selected=b['selected'])
        if (HERE/'primary_baseline.json').exists():assert read(HERE/'primary_baseline.json')==proof
        else:save(HERE/'primary_baseline.json',proof)
    ref=baseline or 'dino_baseline'
    ac={r['index']:r for r in read(COHORT/'dataset_manifest.json') if r['split']=='test'};ix=read(HERE/'config.json')['indices']
    y=np.array([ac[i]['label'] for i in ix]);g=np.array([ac[i]['group_id'] for i in ix]);comparisons={}
    if ref in summary:
        for name,unit in summary.items():
            if name==ref:continue
            delta=np.array(unit['seed_macro_f1'])-np.array(summary[ref]['seed_macro_f1'])
            ci=conditional_bootstrap(probabilities[name],probabilities[ref],y,g)
            gate=float(delta.mean())>.015 and bool((delta>0).all()) and ci['ci95'][0]>0
            comparisons[name]=dict(reference=ref,primary_baseline_frozen=baseline is not None,paired_seed_deltas=delta.tolist(),
                mean_delta=float(delta.mean()),conditional_patient_bootstrap=ci,numerical_gate=gate,
                independent_generalization_confirmed=False,selection_bias_corrected=False)
    report=dict(updated_at=core.timestamp(),test_count=576,test_groups=41,summary=summary,comparisons=comparisons,
        primary_baseline=baseline,temporary_reference=None if baseline else ref,unique_audited_checkpoints=len(units),
        complete_method_groups=len(summary),units=units,test_role='adaptive development feedback',
        historical_test_viewed=True,AC_construction_used_test_labels=True,independent_generalization_confirmed=False)
    save(HERE/'test_results.json',report)
    lines=['# 固定A–C逐方法test反馈记录','',
        '用户授权流程：train训练 → val最早最佳权重 → test测试 → 根据test反馈设计下一方法。test参与开发和选择，以下不是独立泛化确认。历史研究已查看test，A–C构建使用过test标签；等级仅为模型一致性代理。','',
        f'完整方法组：{len(summary)}；独立复算权重：{len(units)}；固定test：576图/41患者。',
        f'固定主要baseline：{baseline or "资格审计尚未结束；DINO仅为临时报告参照"}。','',
        '|方法|seed45 F1|seed46 F1|seed47 F1|平均F1|accuracy|恶性召回|患者等权accuracy|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in summary.items():
        vals=m['seed_macro_f1']+[m[k] for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']]
        lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in vals)+'|')
    lines+=['','所有seed混淆矩阵、checkpoint SHA、预测与审计见test_results.json及predictions/。bootstrap按相同41患者联合重采样、逐seed计算F1再平均，不能消除test反馈选择偏差。完整3seed后才报告方法，无新增训练、无test选epoch。',
        '','## 与固定参照的配对差值','', '|方法|平均差（百分点）|患者条件95%区间（百分点）|','|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    (HERE/'TEST_FEEDBACK_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

def run(watch=False):
    cfg=prepare();cache=read(HERE/'cache_audit.json');assert sha(HERE/'cache/test448.npy')==cache['sha256']
    # A train index must be rejected by this dedicated test loader.
    try:TestData('A',[read(COHORT/'dataset_config.json')['indices']['train'][0]])
    except AssertionError:pass
    else:raise AssertionError('test loader accepted train index')
    while True:
        verify();groups=available();new=[]
        for group in groups:
            p=HERE/'methods'/f'{group["method"]}.json'
            if p.exists():assert read(p)==group
            else:save(p,group);new.append(group['method'])
        if new:safe_commit('research: register complete paired methods and val-selected weight hashes before test feedback')
        for group in groups:
            for u in group['units']:
                print('evaluate/audit',group['method'],u['seed'],flush=True);run_unit(group,u)
        report=summarize(groups)
        save(HERE/'status.json',dict(phase='feedback_ready_continue_research',updated_at=core.timestamp(),
            groups=len(groups),unique_checkpoints=report['unique_audited_checkpoints'],primary_baseline=report['primary_baseline'],
            overall_research_complete=False))
        safe_commit('research: retain all adaptive test feedback, per-image predictions and independent paired audits')
        print('feedback complete',len(groups),report['unique_audited_checkpoints'],flush=True)
        if not watch:return
        if read(Q/'status.json').get('phase') in ['complete','completed','audited'] and report['primary_baseline']:return
        time.sleep(45)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    lock=(HERE/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(args.watch)

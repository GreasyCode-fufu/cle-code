"""Formal CLE field-support round; frozen B0, test-selected mature W0, paired controls."""
from pathlib import Path
import argparse,csv,fcntl,gc,hashlib,json,os,shutil,subprocess,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1]
Q=R/'experiments/ac_qualification_20261024_window01';P=R/'experiments/ac_field_prototype_window01'
T=R/'experiments/ac_test_feedback_window01';FIRST=R/'experiments/ac_structure_20261024_window01'
sys.path[:0]=[str(S),str(Q),str(FIRST),str(R/'experiments'),str(R)]
import run_qualification as q
engine=q.engine;core=q.core
from field_model import FieldModel,counted_flops
from paired_field_data import PairedFieldDataset
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python';GIT=core.GIT
ARMS=['fov_residual','field_pool','reference','no_field','no_local_norm','no_pool'];SEEDS=[45,46,47]
SAVE=core.save_json;READ=core.read_json

def sha(p):return core.digest(p)
def commit(message):
    path=str(S.relative_to(R));staged=subprocess.check_output([GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines()
    assert all(p.startswith(path+'/') for p in staged),'unrelated staged changes; inspect, never consume them'
    subprocess.run([GIT,'-C',str(R),'add','--',path],check=True)
    if subprocess.run([GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([GIT,'-C',str(R),'commit','-m',message],check=True)

def sources():
    paths=[S/n for n in ['run_field_structure.py','audit_field_structure.py','bootstrap_field_structure.py','field_model.py','field_ops_v2.py','paired_field_data.py','FIELD_STRUCTURE_PROTOCOL_ZH.md']]
    paths +=[FIRST/'run_window01.py',Q/'run_qualification.py',Q/'qualification_model.py',R/'experiments/run_data_study.py',T/'audit_test_feedback_v2.py']
    return {str(p.relative_to(R)):sha(p) for p in paths}

def verify():
    cfg=READ(S/'config.json');assert cfg['sources']==sources(),'formal field source drift'
    for p,h in cfg['locked'].items():assert sha(R/p)==h,p
    assert sha(S/'jobs.json')==cfg['jobs_sha256']
    return cfg

def prepare():
    if (S/'config.json').exists():return verify()
    assert os.getuid()==1000 and S.stat().st_uid==1000
    audit=READ(Q/'independent_audit.json');assert audit['passed'] and audit['models']==24
    b0=READ(Q/'selected_baseline.json');assert b0['frozen']
    feedback=READ(T/'test_results.json');assert feedback['complete_method_groups']==15 and feedback['unique_audited_checkpoints']==42 and feedback['primary_baseline']
    assert feedback['primary_baseline']=='qualification_'+b0['selected']['arm']
    options=[(name,m) for name,m in feedback['summary'].items() if name.startswith('qualification_')];assert len(options)==8
    budgets={b['arm']:b for b in READ(Q/'preflight.json')['budgets']}
    name,w0=min(options,key=lambda pair:(-pair[1]['macro_f1'],budgets[pair[0].removeprefix('qualification_')]['counted_flops'],pair[0]))
    arm=name.removeprefix('qualification_');family,recipe=arm.rsplit('_',1)
    e=S/'evidence';e.mkdir(exist_ok=True)
    for source,target in [(T/'test_results.json','test_feedback_snapshot.json'),(Q/'selected_baseline.json','primary_baseline.json'),
            (Q/'independent_audit.json','qualification_audit.json'),(P/'numeric_preflight_v2.json','prototype_numeric.json'),
            (P/'paired_dataset_audit.json','prototype_paired_data.json'),(P/'model_integration_cpu.json','prototype_model_cpu.json')]:
        (e/target).write_bytes(source.read_bytes())
    assert sha(S/'paired_field_data.py')==sha(P/'paired_field_data.py') and sha(S/'field_ops_v2.py')==sha(P/'field_ops_v2.py')
    assert READ(e/'prototype_numeric.json')['passed'] and READ(e/'prototype_paired_data.json')['passed']
    assert READ(e/'prototype_paired_data.json')['full_allowed_tensor_comparisons']==5008
    refs=READ(T/'methods_v2'/f'{name}.json');assert [u['seed'] for u in refs['units']]==SEEDS
    for unit in refs['units']:
        assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
        d=T/'predictions'/unit['fingerprint'];assert READ(d/'independent_audit.json')['passed']
    SAVE(S/'working_reference.json',dict(method=name,selection='maximum complete three-seed test mean among all8 mature qualification arms; adaptive selection disclosed',
        arm=arm,family=family,recipe=recipe,baseline_test=w0,registration=refs,baseline_budget=budgets[arm]))
    cfg=q.verify();(S/'manifest.json').write_bytes((Q/'manifest.json').read_bytes())
    jobs=[dict(id=f'field_{kind}_s{seed}',arm=kind,seed=seed,epochs=60,train=cfg['indices']['train'],select=cfg['indices']['val'],score=cfg['indices']['val']) for kind in ARMS for seed in SEEDS]
    SAVE(S/'jobs.json',jobs)
    locks=[S/'working_reference.json',*e.glob('*.json'),S/'manifest.json',Q/'config.json',Q/'pretrained_manifest.json',Q/'cache448_meta.json',
        R/'experiments/ac_dataset_v1/dataset_config.json',R/'experiments/ac_dataset_v1/dataset_manifest.json',R/'experiments/ac_dataset_v1/independent_dataset_audit.json']
    SAVE(S/'config.json',dict(created_at=core.timestamp(),identity=S.name,sources=sources(),locked={str(p.relative_to(R)):sha(p) for p in locks},
        jobs_sha256=sha(S/'jobs.json'),indices=cfg['indices'],arms=ARMS,seeds=SEEDS,family=family,recipe=recipe,
        primary_baseline=b0['selected']['arm'],working_mature_reference=arm,
        recipe_details=dict(epochs=60,batch_size=32,encoder_lr=1e-5,new_gates_and_head_lr=1e-4,weight_decay=.05,
            optimizer='AdamW',schedule='cosine_to_zero',clip=1.,amp='dynamic_fp16',workers=2,cpu_threads=1,
            augmentation='same RGB/RNG replayed on fixed geometric support',pretraining='same official public initialization as matched W0'),
        data_reuse=dict(prototype_full_rgb_rng_audit=5008,geometry_oracles=160,reason='identical paired data and operator bytes plus locked qualification loader/config; actual CUDA and full427 checks repeated'),
        test_role='adaptive development feedback; not independent confirmation',baseline_retraining=0,new_trainings=18,
        success=dict(mean_macro_f1_delta_strictly_greater_than=.015,all_seed_deltas_positive=True,conditional_patient_CI_lower_positive=True,independent_generalization_confirmed=False)))
    commit('research: freeze CLE geometric-support hypothesis, competitive B0, adaptive mature W0 and paired controls')
    return verify()

def model_for(seed,kind):
    cfg=READ(S/'config.json');assert kind in ['plain',*ARMS]
    base=q.model_for(seed,cfg['family']+'_'+cfg['recipe'])
    return FieldModel(base,cfg['family'],kind).cuda()

def loader(indices,training,seed):
    import torch
    cfg=READ(S/'config.json');recipe=cfg['recipe'];shape=q.RECIPES[recipe]
    data=PairedFieldDataset(q.GuardedData(indices,training,recipe),resize=shape['resize'],crop=shape['crop'],training=training)
    return torch.utils.data.DataLoader(data,batch_size=32,shuffle=training,num_workers=2,pin_memory=True,persistent_workers=True,
        generator=torch.Generator().manual_seed(seed))

def optimizer(model):
    import torch
    enc=[p for name,p in model.encoder.named_parameters() if not name.endswith('.alpha')];ids={id(p) for p in enc}
    rest=[p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([dict(params=enc,lr=1e-5),dict(params=rest,lr=1e-4)],weight_decay=.05)

def signature(job):return hashlib.sha256((json.dumps(job,sort_keys=True)+sha(S/'config.json')).encode()).hexdigest()
def done(job):
    p=S/'runs'/job['id']/'validation_complete.json'
    if not p.exists():return False
    c=READ(p);assert c['signature']==signature(job) and sha(R/c['checkpoint'])==c['checkpoint_sha256']
    assert sha(p.parent/'validation_predictions.csv')==c['prediction_sha256'];return True

def bind():
    engine.STUDY=S;core.STUDY=S;core.WEIGHTS=R/'weights/ac_field_structure_window01'
    engine.verify=verify;engine.source_hashes=sources;engine.model_for=model_for;engine.loader=loader
    engine.optimizer=optimizer;engine.signature=signature;engine.done=done

def worker(job_id):bind();engine.worker(job_id)

def preflight():
    import numpy as np,torch
    from torch.nn import functional as F
    cfg=verify();bind();refs=READ(S/'working_reference.json');budgets=[]
    assert sha(Q/'cache448/images.npy')==READ(Q/'cache448_meta.json')['sha256']
    assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));assert len(x)==32 and len(x[0])==4;x,y=x.cuda(),y.cuda()
    baseline=q.model_for(45,refs['arm']).eval();before={k:v.detach().cpu().clone() for k,v in baseline.state_dict().items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=baseline(x[:,:3].contiguous())
    del baseline;torch.cuda.empty_cache()
    for kind in ['plain',*ARMS]:
        model=model_for(45,kind).eval();torch.cuda.reset_peak_memory_stats()
        for name,value in model.backbone.state_dict().items():
            original=name.replace('.base.','.')
            if original in before:assert torch.equal(value.detach().cpu(),before[original])
            else:assert name.endswith('.alpha')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        assert torch.isfinite(logits).all();inference_peak=torch.cuda.max_memory_allocated()/2**20
        if kind in ['plain','fov_residual','no_field','no_local_norm','no_pool']:assert torch.equal(logits,expected),(kind,float((logits-expected).abs().max()))
        flops=counted_flops(model,x[:1]);opt=optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(model.parameters())) and [g['lr'] for g in opt.param_groups]==[1e-5,1e-4]
        gate_ids={id(p) for n,p in model.named_parameters() if n.endswith('.alpha')}
        assert gate_ids <= {id(p) for p in opt.param_groups[1]['params']}
        n=sum(p.numel() for p in model.parameters());assert n==refs['baseline_budget']['parameters']+len(gate_ids)
        torch.cuda.reset_peak_memory_stats();model.train();scaler=torch.amp.GradScaler('cuda',init_scale=128.);active=0
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            active=max(active,sum(p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.named_parameters() if n.endswith('.alpha')))
            assert any(p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.encoder.named_parameters() if not n.endswith('.alpha'))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        if gate_ids:assert active>0
        assert not torch.equal(model.backbone.head.weight.detach().cpu(),before['head.weight'])
        budgets.append(dict(arm=kind,parameters=n,counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
            inference_peak_gpu_mib=inference_peak,finite_two_real_updates=True,active_gate_gradients=int(active),gate_parameters=len(gate_ids)))
        del model,opt,logits,loss,norm,scaler;gc.collect();torch.cuda.empty_cache()
    del before,x,y,expected;gc.collect();torch.cuda.empty_cache()
    for unit in refs['registration']['units']:
        model=model_for(unit['seed'],'plain');model.backbone.load_state_dict(torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True),strict=True)
        p,y,idx=core.infer(model,loader(cfg['indices']['val'],False,unit['seed']))
        folder=R/unit['complete'];saved=list(csv.DictReader((folder.parent/'validation_predictions.csv').open()))
        expected=np.array([[float(row[f'p{k}']) for k in range(3)] for row in saved]);assert idx==cfg['indices']['val'] and np.max(np.abs(p-expected))<1e-6
        del model,p,expected;torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError as e:assert 'unauthorized split access' in str(e)
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,timestamp=core.timestamp(),budgets=budgets,
        checks=['same original pretrained/head states','batch32 CUDA AMP two finite updates','nonzero gates and encoder gradients','exact zero gate logits',
            'full427 matched W0 checkpoint probabilities for all3 seeds','optimizer exact coverage/gate lr','fixed train/val source and RNG audit reuse','test rejection'],
        flops_excludes='normalization/activation/softmax/pooling/elementwise and data movement; explicit functional convolution/attention included'))
    commit('test: validate formal geometric-support batch32 CUDA AMP, exact baseline reuse and finite operators')

def bench(label):
    import torch
    from torch.nn import functional as F
    phase,index=map(int,label.split(':'));bind();cfg=verify();model=model_for(45+index%3,'fov_residual');opt=optimizer(model)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.);data=loader(cfg['indices']['train'],True,45);it=iter(data);model.train()
    def step():
        nonlocal it
        start=time.perf_counter()
        try:x,y,_,_=next(it)
        except StopIteration:it=iter(data);x,y,_,_=next(it)
        wait=time.perf_counter()-start;x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update();torch.cuda.synchronize();return len(y),wait
    for _ in range(4):step()
    folder=S/'profile';SAVE(folder/f'ready_{phase}_{index}.json',dict(pid=os.getpid()))
    gate=folder/f'gate_{phase}.json';deadline=time.monotonic()+240
    while not gate.exists():assert time.monotonic()<deadline;time.sleep(.05)
    start=READ(gate)['start_unix']
    while time.time()<start:time.sleep(.01)
    n=0;wait=0.;tick=time.perf_counter()
    while time.time()<start+12:
        amount,w=step();n+=amount;wait+=w
    elapsed=time.perf_counter()-tick
    SAVE(folder/f'bench_{phase}_{index}.json',dict(samples=n,seconds=elapsed,samples_per_second=n/elapsed,loader_wait_seconds=wait,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))

def profile():
    cfg=verify();folder=S/'profile';folder.mkdir(exist_ok=True);results=[]
    foot=max(2048,next(x['peak_gpu_mib'] for x in READ(S/'preflight.json')['budgets'] if x['arm']=='fov_residual')*1.25+700)
    ceiling=max(1,min(os.cpu_count()//2,int((engine.snapshot()['gpu_free_mib']-6144)/foot)))
    tested=sorted(set([1,min(2,ceiling),min(4,ceiling)]))
    for slots in tested:
        active=[]
        for index in range(slots):
            log=(folder/f'bench_{slots}_{index}.log').open('w');p=subprocess.Popen([PY,__file__,'bench','--job',f'{slots}:{index}'],cwd=R,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);active.append((p,log,index))
        deadline=time.monotonic()+240
        while not all((folder/f'ready_{slots}_{i}.json').exists() for _,_,i in active):
            assert all(p.poll() is None for p,_,_ in active),'benchmark initialization failed; inspect exact logs'
            assert time.monotonic()<deadline;time.sleep(.2)
        SAVE(folder/f'gate_{slots}.json',dict(start_unix=time.time()+.5));stats=[]
        while any(p.poll() is None for p,_,_ in active):
            x=engine.snapshot();stats.append(x);engine.append_resource(folder/'resources.csv',dict(**x,slots=slots));time.sleep(2)
        for p,log,_ in active:log.close();assert p.returncode==0
        measures=[READ(folder/f'bench_{slots}_{i}.json') for _,_,i in active]
        result=dict(slots=slots,aggregate_samples_per_second=sum(x['samples_per_second'] for x in measures),workers=measures,
            mean_gpu_util=sum(x['gpu_util'] for x in stats)/len(stats),max_gpu_used_mib=max(x['gpu_used_mib'] for x in stats));results.append(result);SAVE(folder/'measurements.json',results)
        if slots>=2 and result['mean_gpu_util']>=98 and result['aggregate_samples_per_second']<results[-2]['aggregate_samples_per_second']*1.03:break
        if slots==tested[-1] and slots<ceiling and result['mean_gpu_util']<80 and (len(results)==1 or result['aggregate_samples_per_second']>results[-2]['aggregate_samples_per_second']*1.03):tested.append(min(ceiling,slots+2))
    best=max(x['aggregate_samples_per_second'] for x in results);selected=next(x['slots'] for x in results if x['aggregate_samples_per_second']>=.97*best)
    SAVE(folder/'selected.json',dict(initial_slots=selected,measured_slots=[x['slots'] for x in results],memory_ceiling=ceiling,rule='smallest within97percent of best measured aggregate throughput'))
    commit('perf: measure actual geometric-support single and concurrent training throughput')

def run():
    verify();bind();assert READ(S/'preflight.json')['passed']
    with (S/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=READ(S/'jobs.json');pending=[j for j in jobs if not done(j)];completed={j['id'] for j in jobs if j not in pending}
        bs=READ(S/'preflight.json')['budgets'];footprints={x['arm']:max(2048,x['peak_gpu_mib']*1.25+700) for x in bs}
        infer=max(x['inference_peak_gpu_mib']*1.25+700 for x in bs)
        capacity=engine.snapshot()['gpu_free_mib']-6144;ceiling=min(os.cpu_count()//2,len(jobs),max(1,int(capacity/min(footprints.values()))))
        slots=min(ceiling,READ(S/'profile/selected.json')['initial_slots']);running={};faults=[];recent=[];next_sample=0
        logs=S/'logs';logs.mkdir(exist_ok=True);SAVE(S/'pipeline_pid.json',dict(pid=os.getpid(),identity=S.name,started_at=core.timestamp()))
        from audit_field_structure import evaluate_completed_groups
        while pending or running:
            changes=[]
            for pid,(p,j,log) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[pid]
                if p.returncode or not done(j):faults.append(dict(job=j['id'],returncode=p.returncode,log=str(logs/(j['id']+'.log'))))
                else:completed.add(j['id']);changes.append(j['id'])
            if time.monotonic()>=next_sample:
                stats=engine.snapshot();recent.append(stats);recent=recent[-15:];engine.append_resource(S/'resources.csv',dict(**stats,running=len(running),pending=len(pending),slots=slots))
                if len(recent)==15 and pending and len(running)>=slots and sum(x['gpu_util'] for x in recent)/15<80 and stats['cpu_percent']<75:slots=min(slots+1,ceiling)
                next_sample=time.monotonic()+2
            if changes:commit('results: retain complete field-support training '+','.join(changes))
            if stats['gpu_free_mib']>6144+infer:evaluate_completed_groups()
            while pending and len(running)<slots:
                reserved=sum(footprints[j['arm']] for _,j,_ in running.values())
                pos=next((i for i,j in enumerate(pending) if reserved+footprints[j['arm']]<=capacity and stats['gpu_free_mib']>6144+footprints[j['arm']]),None)
                if pos is None or stats['cpu_percent']>92:break
                j=pending.pop(pos);log=(logs/(j['id']+'.log')).open('a',buffering=1)
                p=subprocess.Popen([PY,__file__,'worker','--job',j['id']],cwd=R,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                running[p.pid]=(p,j,log);print('launch',j['id'],p.pid,flush=True)
            SAVE(S/'status.json',dict(phase='training' if pending or running else 'needs_final_round_audit',completed_new=len(completed),total_new=18,
                pending=len(pending),running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()],failures=faults,slots=slots,resources=stats,updated_at=core.timestamp(),overall_research_complete=False))
            if pending or running:time.sleep(.5)
        assert not faults,'Inspect precise training logs and live pipeline; complete results must not be retrained'
        evaluate_completed_groups();subprocess.run([PY,str(S/'audit_field_structure.py'),'--final'],cwd=R,env=engine.thread_env(),check=True)
        SAVE(S/'status.json',dict(phase='field_round_audited_continue_mechanism_research',completed_new=18,updated_at=core.timestamp(),overall_research_complete=False))
        commit('results: retain full geometric-support round, adaptive test feedback and negative/positive controls')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','bench','profile','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='worker':worker(args.job)
    elif args.command=='bench':bench(args.job)
    else:globals()[args.command]()

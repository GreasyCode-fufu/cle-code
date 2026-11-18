"""New independent library/CLE mechanism study; train-only gradients, val-best, adaptive test feedback."""
from pathlib import Path
import argparse,csv,fcntl,gc,hashlib,json,os,subprocess,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1]
Q=R/'experiments/ac_qualification_20261024_window01';T=R/'experiments/ac_test_feedback_window01';F=R/'experiments/ac_field_structure_window01'
M=R/'models/optimize/ac_channel_covariance_window01';PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
sys.path[:0]=[str(S),str(Q),str(R/'experiments/ac_structure_20261024_window01'),str(R/'experiments'),str(R)]
import run_qualification as q
import run_window01 as engine
from models.optimize.ac_channel_covariance_window01.library_cle import LibraryNet,ARMS,counted_flops
from qualification_model import counted_flops as base_flops
core=q.core;SAVE=core.save_json;READ=core.read_json;sha=core.digest
SEEDS=[45,46,47];ACTIVE_ARM='plain'

def sources():
    names=['run_module_structure.py','audit_module_structure.py','check_module_sources.py','bootstrap_module_structure.py','MODULE_PROTOCOL_ZH.md','MODULE_DIFFERENCES_ZH.md','build_covariance_study.py','preflight_impl.py','run_covariance_manager.py','continuous_runtime_sampler.py','compare_covariance_controls.py','finish_direct_controls.py','independent_probability_auditor.py','predict_module_model.py','check_prediction_cli.py']
    paths=[S/n for n in names]+sorted(M.glob('*.py'))
    paths += [Q/'run_qualification.py',Q/'qualification_model.py',R/'experiments/ac_structure_20261024_window01/run_window01.py',R/'experiments/run_data_study.py']
    return {str(p.relative_to(R)):sha(p) for p in paths if p.is_file()}

def commit(message):
    allowed=[str(S.relative_to(R)),str(M.relative_to(R))]
    staged=subprocess.check_output([core.GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines()
    assert all(any(p.startswith(a+'/') for a in allowed) for p in staged),'unrelated staged changes; do not alter or commit them'
    subprocess.run([core.GIT,'-C',str(R),'add','--',*allowed],check=True)
    if subprocess.run([core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([core.GIT,'-C',str(R),'commit','-m',message],check=True)

def verify():
    q.verify();cfg=READ(S/'config.json');assert cfg['sources']==sources(),'module source drift'
    assert sha(S/'jobs.json')==cfg['jobs_sha256']
    for p,h in cfg['locked'].items():assert sha(R/p)==h,p
    return cfg

def prepare():
    if (S/'config.json').exists():return verify()
    assert os.getuid()==1000 and all(p.stat().st_uid==1000 for root in [S,M] for p in [root,*root.rglob('*')])
    q.verify();d=READ(R/'experiments/ac_dataset_v1/dataset_config.json');baseline=READ(Q/'selected_baseline.json')
    assert 'convnext_tiny_A' in json.dumps(baseline) and READ(Q/'independent_audit.json')['passed']
    (S/'manifest.json').write_bytes((Q/'manifest.json').read_bytes());(S/'baseline_selection.json').write_bytes((Q/'selected_baseline.json').read_bytes())
    refs=READ(T/'methods_v2/qualification_convnext_tiny_A.json');assert [u['seed'] for u in refs['units']]==SEEDS;SAVE(S/'baseline_reference.json',refs)
    jobs=[dict(id=f'cov_{arm}_s{seed}',arm=arm,seed=seed,epochs=60,train=d['indices']['train'],select=d['indices']['val'],score=d['indices']['val']) for arm in ARMS for seed in SEEDS];SAVE(S/'jobs.json',jobs)
    locked=[R/'experiments/ac_dataset_v1'/n for n in ['dataset_config.json','dataset_manifest.json','independent_dataset_audit.json']]
    locked += [S/n for n in ['manifest.json','shared_snapshot_manifest.json','baseline_reference.json','baseline_selection.json','working_reference.json','postprocessing_revision_v2.json']]
    for stage,names in [('ac_channel_covariance_prototype_window01',['config.json','numeric_preflight_cpu.json','complete.json']),('ac_spatial_lag_structure_window01',['config.json','independent_audit.json','direct_control_comparisons.json']),('ac_module_structure_window01',['config.json','independent_audit.json'])]:
        locked += [R/'experiments'/stage/n for n in names]
    locked += [Q/'config.json',Q/'pretrained_manifest.json',Q/'independent_audit.json',T/'cache_audit.json',R/'.cache_data_study/meta.json']
    SAVE(S/'config.json',dict(created_at=core.timestamp(),identity=S.name,thread='01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76',sources=sources(),locked={str(p.relative_to(R)):sha(p) for p in locked},jobs_sha256=sha(S/'jobs.json'),indices=d['indices'],seeds=SEEDS,arms=ARMS,baseline='qualification_convnext_tiny_A',family='convnext_tiny',recipe='A',
        recipe_details=dict(epochs=60,batch=32,resize=256,crop=224,encoder_lr=1e-5,new_module_and_head_lr=1e-4,weight_decay=.05,optimizer='AdamW',schedule='cosine_to_zero',amp='dynamic_fp16',clip=1,stochastic_depth=.1,workers=2,cpu_threads=1),
        descriptor=dict(location='same last ConvNeXt feature Bx768x7x7 before original average pool',project='biasless learned768to16 1x1',covariance='spatially centered XXtranspose/HW',ridge=1e-4,square_root='trace prenormalized5step Newton Schulz,3BMM each iteration,postmultiply sqrt(trace)',floor='subtract cached same5iteration sqrt(ridgeI); fixed translation',vector='136 upper triangular coordinates,offdiagonal sqrt2 Frobenius weighting',normalization='no-affine LN eps1e-5',zero_classifier='136to3 logits added to unchanged GAP learnednorm head',extra_parameters=12699,retained_moment_parameters=18528,rawcov='direct covariance with same projection/readout',diag='clear offdiagonal before same5iterations;136nominal coordinates but only16effective,not equal effective capacity',channelperm='independent fixed49position permutation per projected channel preserving marginals and changing joint pairing',gradient_rule='classifier firstgradient positive; upstream project firstgradient zero and secondgradient positive; retained Mom firstgradient positive',original_CE_only=True),
        weight_initialization='same official ImageNet and seeded head as original M/Q; fork RNG for new operators; no CLE warmstart',checkpoint_selection='earliest maximum original0degree val macroF1 across60 epochs',training_data='2077 unique fixed AC indices per epoch; physical3595 cache is not training3595; no relabel/grade/split/filter/resample changes',test_role='adaptive internal development by explicit authorization; historical test viewed and AC construction used labels; conditional CI not selection-adjusted',
        success=dict(mean_macro_f1_delta_strictly_greater_than=.015,all_paired_seed_deltas_positive=True,conditional_patient_ci_lower_positive=True,independent_generalization_confirmation=False),bootstrap=dict(replicates=20000,seed=20261024),novelty='Same-layer covariance and iterative matrix square root are mature iSQRT-COV concepts; projection ridge floor vector LN GAP residual adaptation is not full official reproduction or automatic novelty; no SCI acceptance or gain guarantee',overall_research_complete=False))
    SAVE(S/'comparison_preregistration.json',dict(registered_at=core.timestamp(),source_config_sha256=sha(S/'config.json'),comparator_sha256=sha(S/'compare_covariance_controls.py'),finisher_sha256=sha(S/'finish_direct_controls.py'),comparisons=['mom_covsqrt vs original M moments','mom_covsqrt vs covsqrt_only','mom_covsqrt vs mom_channelperm','mom_covsqrt vs mom_diag','mom_covsqrt vs mom_rawcov','mom_covsqrt vs fixedB0'],modes=['single','c4'],seeds=SEEDS,registered_before_training=True,threshold='Fraction mean >3/200, allseed positive,41patient conditionalCI lower positive',independent_generalization=False))
    commit('research: preregister retained CNN moment plus same-layer covariance power readout with true drop-Mom raw covariance diagonal and joint-pairing controls')
    return verify()

def model_for(seed,arm):
    assert arm in ['plain','mom_only',*ARMS]
    return LibraryNet(q.model_for(seed,'convnext_tiny_A'),arm).cuda()

class Data:
    def __init__(self,indices,training,seed,arm):
        self.base=q.GuardedData(indices,training,'A');self.rotate=False;self.rng=None
    def __len__(self):return len(self.base)
    def __getitem__(self,pos):
        import torch
        x,y,w,i=self.base[pos]
        if self.rotate:
            if self.rng is None:self.rng=torch.Generator().manual_seed((torch.initial_seed()+0xC4000000)%(2**63-1))
            k=int(torch.randint(4,(),generator=self.rng));x=torch.rot90(x,k,(-2,-1)).contiguous()
        return x,y,w,i

def loader(indices,training,seed):
    import torch
    return torch.utils.data.DataLoader(Data(indices,training,seed,ACTIVE_ARM),batch_size=32,shuffle=training,num_workers=2,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed))

def optimizer(model):
    import torch
    enc=[p for p in model.encoder.parameters() if id(p) in model.original_parameter_ids];ids={id(p) for p in enc}
    rest=[p for p in model.parameters() if id(p) not in ids]
    assert all(p.requires_grad for p in model.parameters())
    return torch.optim.AdamW([dict(params=enc,lr=1e-5),dict(params=rest,lr=1e-4)],weight_decay=.05)

def signature(job):return hashlib.sha256((json.dumps(job,sort_keys=True)+sha(S/'config.json')).encode()).hexdigest()
def done(job):
    p=S/'runs'/job['id']/'validation_complete.json'
    if not p.exists():return False
    c=READ(p);assert c['signature']==signature(job) and sha(R/c['checkpoint'])==c['checkpoint_sha256']
    assert sha(p.parent/'validation_predictions.csv')==c['prediction_sha256'];return True
def bind():
    engine.STUDY=S;core.STUDY=S;core.WEIGHTS=R/'weights/ac_channel_covariance_structure_window01'
    engine.verify=verify;engine.source_hashes=sources;engine.model_for=model_for;engine.loader=loader;engine.optimizer=optimizer;engine.signature=signature;engine.done=done
def worker(job_id):
    global ACTIVE_ARM
    job=next(j for j in READ(S/'jobs.json') if j['id']==job_id);ACTIVE_ARM=job['arm'];bind();engine.worker(job_id)

def preflight():
    import numpy as np,torch
    from torch.nn import functional as functional
    cfg=verify();bind();assert READ(S/'module_source_audit.json')['passed'];assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    cache_audit=READ(T/'cache_audit.json');assert cache_audit['passed'] and sha(T/'cache/test448.npy')==cache_audit['sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));x,y=x.cuda(),y.cuda();plain=model_for(45,'plain').eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=plain(x).clone()
    del plain;torch.cuda.empty_cache();budgets=[];diagnostics=[]
    for arm in ['plain','mom_only',*ARMS]:
        model=model_for(45,arm).eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pred=model(x)
        assert torch.equal(pred,expected),arm
        opt=optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()));flops=counted_flops(model,x[:1],base_flops);model.train();torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda',init_scale=128.);gradients=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=functional.cross_entropy(model(x),y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),arm
            assert any(p.grad is not None and float(p.grad.norm())>0 for p in opt.param_groups[0]['params'])
            if arm in ARMS:
                classifier_norm=float(model.readout.classifier.weight.grad.norm());project_norm=float(model.readout.project.weight.grad.norm())
                assert classifier_norm>0 and (project_norm==0. if step==0 else project_norm>0), (arm,step,classifier_norm,project_norm)
                moment_norm=float(model.encoder[0][0][0].adapter.weight.grad.norm()) if arm!='covsqrt_only' else None
                if moment_norm is not None:assert moment_norm>0
                gradients.append(dict(step=step,CE_loss=float(loss.detach()),zero_classifier_gradient_norm=classifier_norm,upstream_project_gradient_norm=project_norm,retained_moment_gradient_norm=moment_norm,expected_upstream_first_zero_second_positive=True))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        if arm in ARMS:diagnostics.append(dict(arm=arm,gradients=gradients,actual_train_images=len(idx),test_images=0))
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_two_updates=True));del model,opt,pred,scaler,loss;gc.collect();torch.cuda.empty_cache()
    assert len(idx)==32 and len(set(idx.tolist()))==32 and set(idx.tolist())<=set(cfg['indices']['train'])
    del x,y,expected;reproduced=[]
    for kind,units in [('plain',READ(S/'baseline_reference.json')['units']),('mom_only',[u for u in READ(S/'working_reference.json')['units'] if u['arm']=='c4_moments'])]:
        for unit in units:
            assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];model=model_for(unit['seed'],kind);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True);(model.backbone if kind=='plain' else model).load_state_dict(state,strict=True);del state
            p,yy,ii=core.infer(model,loader(cfg['indices']['val'],False,unit['seed']));saved=list(csv.DictReader(((R/unit['complete']).parent/'validation_predictions.csv').open()));target=np.array([[float(z[f'p{k}']) for k in range(3)] for z in saved]);error=float(np.max(np.abs(p-target)));assert ii==cfg['indices']['val'] and error==0.;reproduced.append(dict(kind=kind,seed=unit['seed'],full427_max_probability_error=error));del model;gc.collect();torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError:pass
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,budgets=budgets,descriptor_diagnostics=diagnostics,reproduced_baselines=reproduced,test_denied=True,objective='original CE only; patient auxiliary targets not retained as confirmed increments',flops_convention='2FLOP/MAC includes3 functional rotated convolutions,new Conv/Linear,covariance BMM and5iterations each3BMM; cached constructor floor excluded; excludes centering/normalization/pointwise/memory, actual throughput separately measured'))
    commit('test: verify channel covariance controls CUDA AMP two-step chain-rule gradients pairing and all6 frozen full427 probability reconstructions')






def bench(label):
    global ACTIVE_ARM
    import torch
    from torch.nn import functional as functional
    arm,slots,index=label.split(':');ACTIVE_ARM=arm;index=int(index);bind();cfg=verify();model=model_for(45+index%3,arm);opt=optimizer(model);model.train();scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    dl=loader(cfg['indices']['train'],True,45+index);it=iter(dl)
    def step():
        nonlocal it
        start=time.perf_counter()
        try:x,y,_,_=next(it)
        except StopIteration:it=iter(dl);x,y,_,_=next(it)
        wait=time.perf_counter()-start;x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=functional.cross_entropy(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();torch.cuda.synchronize();return len(y),wait
    for _ in range(4):step()
    folder=S/'profile'/arm;SAVE(folder/f'ready_{slots}_{index}.json',dict(pid=os.getpid()))
    gate=folder/f'gate_{slots}.json';deadline=time.monotonic()+300
    while not gate.exists():assert time.monotonic()<deadline;time.sleep(.05)
    start=READ(gate)['start_unix']
    while time.time()<start:time.sleep(.01)
    n=0;wait=0.;tick=time.perf_counter()
    while time.time()<start+12:amount,w=step();n+=amount;wait+=w
    elapsed=time.perf_counter()-tick;SAVE(folder/f'bench_{slots}_{index}.json',dict(samples=n,seconds=elapsed,samples_per_second=n/elapsed,loader_wait_seconds=wait,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))

def profile():
    cfg=verify();selected={};budgets={b['arm']:b for b in READ(S/'preflight.json')['budgets']}
    # Profile every complete spatial-statistic objective; do not inherit another study concurrency.
    for arm in ARMS:
        folder=S/'profile'/arm;folder.mkdir(parents=True,exist_ok=True);results=[]
        footprint=budgets[arm]['peak_gpu_mib']*1.25+700;ceiling=max(1,int((engine.snapshot()['gpu_free_mib']-6144)/footprint))
        tested=sorted(set([1,min(2,ceiling),min(4,ceiling)]))
        for slots in tested:
            active=[]
            for i in range(slots):
                log=(folder/f'bench_{slots}_{i}.log').open('a');p=subprocess.Popen([PY,__file__,'bench','--job',f'{arm}:{slots}:{i}'],cwd=R,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);active.append((p,log,i))
            deadline=time.monotonic()+300
            while not all((folder/f'ready_{slots}_{i}.json').exists() for _,_,i in active):assert all(p.poll() is None for p,_,_ in active) and time.monotonic()<deadline,'profile startup failure';time.sleep(.2)
            gate=time.time()+.5;SAVE(folder/f'gate_{slots}.json',dict(start_unix=gate));stats=[]
            while any(p.poll() is None for p,_,_ in active):
                st=engine.snapshot();engine.append_resource(folder/'resources.csv',dict(**st,slots=slots,unix=time.time()))
                if gate+1<time.time()<gate+11:stats.append(st)
                time.sleep(1)
            for p,log,_ in active:log.close();assert p.returncode==0,'profile failed; inspect exact log'
            measures=[READ(folder/f'bench_{slots}_{i}.json') for _,_,i in active]
            row=dict(slots=slots,aggregate_samples_per_second=sum(x['samples_per_second'] for x in measures),workers=measures,mean_active_gpu_util=sum(x['gpu_util'] for x in stats)/len(stats),mean_active_cpu_util=sum(x['cpu_percent'] for x in stats)/len(stats));results.append(row);SAVE(folder/'measurements.json',results)
            if len(results)>1 and row['mean_active_gpu_util']>=98 and row['aggregate_samples_per_second']<results[-2]['aggregate_samples_per_second']*1.03:break
            if slots==tested[-1] and slots<ceiling and row['mean_active_gpu_util']<80 and row['mean_active_cpu_util']<85 and (len(results)==1 or row['aggregate_samples_per_second']>results[-2]['aggregate_samples_per_second']*1.03):tested.append(min(ceiling,slots+2))
        best=max(x['aggregate_samples_per_second'] for x in results);selected[arm]=next(x['slots'] for x in results if x['aggregate_samples_per_second']>=.97*best)
    SAVE(S/'profile/selected.json',dict(slots_by_arm=selected,rule='smallest within97percent of measured best; active-kernel GPU samples, not interpreter startup;6GiB reserve'))
    commit('perf: measure single and concurrent throughput for distinct library and orientation kernels')

def run():
    cfg=verify();bind();assert READ(S/'preflight.json')['passed'];policy=READ(S/'profile/selected.json')['slots_by_arm']
    with (S/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=READ(S/'jobs.json');pending=[j for j in jobs if not done(j)];completed={j['id'] for j in jobs if done(j)};running={};faults=[]
        SAVE(S/'manager_identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__),started_at=core.timestamp()))
        logs=S/'logs';logs.mkdir(exist_ok=True);budget={b['arm']:b for b in READ(S/'preflight.json')['budgets']}
        from audit_module_structure import evaluate_completed_groups,final_audit
        tick=0
        while pending or running:
            changes=[]
            for pid,(p,j,log) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[pid]
                if p.returncode or not done(j):faults.append(dict(job=j['id'],pid=pid,returncode=p.returncode,log=str(log.name)))
                else:completed.add(j['id']);changes.append(j['id'])
            if changes:commit('results: preserve completed module training '+','.join(changes))
            if not running:evaluate_completed_groups()
            stats=engine.snapshot()
            if pending and not faults:
                active_arm=next(iter(running.values()))[1]['arm'] if running else pending[0]['arm']
                while len(running)<policy[active_arm]:
                    pos=next((i for i,j in enumerate(pending) if j['arm']==active_arm),None)
                    if pos is None:break
                    if stats['gpu_free_mib']<6144+budget[active_arm]['peak_gpu_mib']*1.25+700 or stats['cpu_percent']>92:break
                    j=pending.pop(pos);log=(logs/(j['id']+'.log')).open('a',buffering=1);p=subprocess.Popen([PY,__file__,'worker','--job',j['id']],cwd=R,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);running[p.pid]=(p,j,log)
                    print('launch',j['id'],p.pid,flush=True)
            if time.monotonic()>tick:
                engine.append_resource(S/'resources.csv',dict(**stats,running=len(running),pending=len(pending)))
                # Include diskstats so I/O changes can be reconstructed independently.
                with (S/'io_resources.jsonl').open('a') as file:file.write(json.dumps(dict(timestamp=core.timestamp(),diskstats=Path('/proc/diskstats').read_text()))+'\n')
                tick=time.monotonic()+2
            SAVE(S/'status.json',dict(phase='training' if running or pending else 'audit',completed_new=len(completed),total_new=len(jobs),pending=len(pending),running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()],failures=faults,resources=stats,updated_at=core.timestamp(),overall_research_complete=False))
            if faults and not running:raise RuntimeError('precise own training failure; preserve logs and complete checkpoints, do not blind retrain')
            if pending or running:time.sleep(.5)
        evaluate_completed_groups();final_audit();SAVE(S/'status.json',dict(phase='module_round_audited_continue_mechanism_research',completed_new=len(completed),total_new=len(jobs),overall_research_complete=False,updated_at=core.timestamp()))
        commit('results: audit complete faithful-module and orientation round; retain every seed and negative result')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','profile','bench','worker','run']);p.add_argument('--job');a=p.parse_args()
    if a.command=='worker':worker(a.job)
    elif a.command=='bench':bench(a.job)
    else:globals()[a.command]()

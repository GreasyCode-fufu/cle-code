"""Measured-throughput admission; adopts existing workers without restarting models."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f

def command(pid):
    try:
        proc=Path('/proc')/str(pid)
        if (proc/'stat').read_text().split(') ',1)[1].startswith('Z '):return ''
        return (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,PermissionError,ProcessLookupError):return ''

def group_for(arm):return arm if arm in ['field_pool','reference'] else 'fov_residual'

def choose(arm):
    group=group_for(arm)
    if group=='fov_residual':return f.READ(S/'profile/selected.json')['initial_slots']
    folder=S/'runtime_profiles'/group
    if (folder/'selected.json').exists():return f.READ(folder/'selected.json')['initial_slots']
    folder.mkdir(parents=True,exist_ok=True);budget=next(b for b in f.READ(S/'preflight.json')['budgets'] if b['arm']==arm)
    foot=max(2048,budget['peak_gpu_mib']*1.25+700);ceiling=max(1,min(os.cpu_count()//2,int((f.engine.snapshot()['gpu_free_mib']-6144)/foot)))
    candidates=sorted(set([1,min(2,ceiling),min(4,ceiling)]));results=[]
    for slots in candidates:
        active=[]
        for index in range(slots):
            log=(folder/f'bench_{slots}_{index}.log').open('w');proc=subprocess.Popen([f.PY,str(S/'bench_field_runtime.py'),'--arm',arm,'--slots',str(slots),'--index',str(index)],
                cwd=f.R,env=f.engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);active.append((proc,log,index))
        deadline=time.monotonic()+240
        while not all((folder/f'ready_{slots}_{i}.json').exists() for _,_,i in active):
            assert all(proc.poll() is None for proc,_,_ in active),'Runtime profile failed; inspect benchmark logs, do not train from its state'
            assert time.monotonic()<deadline;time.sleep(.2)
        f.SAVE(folder/f'gate_{slots}.json',dict(start_unix=time.time()+.5));stats=[]
        while any(proc.poll() is None for proc,_,_ in active):
            x=f.engine.snapshot();stats.append(x);f.engine.append_resource(folder/'resources.csv',dict(**x,slots=slots));time.sleep(2)
        for proc,log,_ in active:log.close();assert proc.returncode==0
        measures=[f.READ(folder/f'bench_{slots}_{i}.json') for _,_,i in active]
        current=dict(slots=slots,aggregate_samples_per_second=sum(m['samples_per_second'] for m in measures),workers=measures,
            mean_gpu_util=sum(x['gpu_util'] for x in stats)/len(stats));results.append(current);f.SAVE(folder/'measurements.json',results)
        if slots>=2 and current['mean_gpu_util']>=98 and current['aggregate_samples_per_second']<results[-2]['aggregate_samples_per_second']*1.03:break
        if slots==candidates[-1] and slots<ceiling and current['mean_gpu_util']<80 and current['aggregate_samples_per_second']>results[-2]['aggregate_samples_per_second']*1.03:candidates.append(min(ceiling,slots+2))
    best=max(m['aggregate_samples_per_second'] for m in results);selected=next(m['slots'] for m in results if m['aggregate_samples_per_second']>=.97*best)
    f.SAVE(folder/'selected.json',dict(initial_slots=selected,rule='smallest within97percent best measured aggregate, no startup-utilization override',arm=arm,discarded_benchmark_states=True))
    f.commit('perf: select '+arm+' admission from actual single/concurrent throughput');return selected

def main():
    cfg=f.verify();f.bind();assert f.READ(S/'preflight.json')['passed']
    policy=f.READ(S/'queue_policy_v2.json');assert policy['config_sha256']==f.sha(S/'config.json')
    assert policy['source_sha256']=={n:f.sha(S/n) for n in policy['source_sha256']}
    with (S/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        transition=f.READ(S/'QUEUE_TRANSITION_V2.json');jobs=f.READ(S/'jobs.json');jobmap={j['id']:j for j in jobs};running={};completed=set();faults=[]
        for item in transition['adopt_workers']:
            job=jobmap[item['job']];cmd=command(item['pid'])
            if cmd:
                assert 'ac_field_structure_window01/run_field_structure.py worker --job '+item['job'] in cmd
                running[item['pid']]=dict(job=job,process=None,log=None,adopted=True);print('adopt existing',item['job'],item['pid'],flush=True)
            elif f.done(job):completed.add(job['id'])
            else:raise RuntimeError('Adopted worker absent without valid completion: inspect existing log, no automatic retraining')
        pending=[j for j in jobs if j['id'] not in completed and j['id'] not in {x['job']['id'] for x in running.values()} and not f.done(j)]
        completed |= {j['id'] for j in jobs if j not in pending and j['id'] not in {x['job']['id'] for x in running.values()}}
        budgets=f.READ(S/'preflight.json')['budgets'];feet={b['arm']:max(2048,b['peak_gpu_mib']*1.25+700) for b in budgets};infer=max(b['inference_peak_gpu_mib']*1.25+700 for b in budgets)
        f.SAVE(S/'queue_manager_identity.json',dict(pid=os.getpid(),source=str(Path(__file__).resolve()),source_sha256=f.sha(__file__),started_at=f.core.timestamp(),policy_sha256=f.sha(S/'queue_policy_v2.json')))
        f.SAVE(S/'pipeline_pid_v2.json',dict(pid=os.getpid(),source='manage_field_queue_v2.py'))
        from audit_field_structure import evaluate_completed_groups
        next_sample=0;stats=f.engine.snapshot();slots=choose('fov_residual')
        while pending or running:
            changes=[]
            for pid,item in list(running.items()):
                proc=item['process'];alive=(proc.poll() is None) if proc is not None else bool(command(pid))
                if alive:continue
                if item['log'] is not None:item['log'].close()
                del running[pid];job=item['job']
                if (proc is not None and proc.returncode) or not f.done(job):faults.append(dict(job=job['id'],pid=pid,log=str(S/'logs'/(job['id']+'.log')),returncode=proc.returncode if proc else None))
                else:completed.add(job['id']);changes.append(job['id'])
            if time.monotonic()>=next_sample:
                stats=f.engine.snapshot();f.engine.append_resource(S/'queue_resources_v2.csv',dict(**stats,running=len(running),pending=len(pending),measured_slots=slots));next_sample=time.monotonic()+2
            if changes:f.commit('results: retain completed field training '+','.join(changes))
            if stats['gpu_free_mib']>6144+infer:evaluate_completed_groups()
            if pending and not running:slots=choose(pending[0]['arm'])
            while pending and len(running)<slots:
                job=pending[0]
                # A new compute family is measured in isolation after current workers finish.
                if any(group_for(x['job']['arm'])!=group_for(job['arm']) for x in running.values()):break
                if not running:slots=choose(job['arm'])
                if len(running)>=slots:break
                stats=f.engine.snapshot()
                if stats['gpu_free_mib']<6144+feet[job['arm']] or stats['cpu_percent']>92:break
                pending.pop(0);log=(S/'logs'/(job['id']+'.log')).open('a',buffering=1)
                proc=subprocess.Popen([f.PY,str(S/'run_field_structure.py'),'worker','--job',job['id']],cwd=f.R,env=f.engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                running[proc.pid]=dict(job=job,process=proc,log=log,adopted=False);print('launch measured admission',job['id'],proc.pid,'slots',slots,flush=True)
            f.SAVE(S/'queue_status_v2.json',dict(phase='training',completed_new=len(completed),total_new=18,pending=len(pending),
                running=[dict(job=x['job']['id'],pid=pid,adopted=x['adopted']) for pid,x in running.items()],failures=faults,measured_slots=slots,
                startup_utilization_does_not_override_throughput=True,updated_at=f.core.timestamp(),overall_research_complete=False))
            if pending or running:time.sleep(.5)
        assert not faults,'Inspect exact worker failures and valid checkpoints; no blind retraining'
        evaluate_completed_groups();subprocess.run([f.PY,str(S/'audit_field_structure.py'),'--final'],cwd=f.R,env=f.engine.thread_env(),check=True)
        f.SAVE(S/'queue_status_v2.json',dict(phase='field_round_audited_continue_research',completed_new=18,overall_research_complete=False,updated_at=f.core.timestamp()))
        f.commit('results: finish formal field round under measured-throughput admission and retain every outcome')

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        f.SAVE(S/'queue_fault_v2.json',dict(timestamp=f.core.timestamp(),traceback=traceback.format_exc()))
        raise

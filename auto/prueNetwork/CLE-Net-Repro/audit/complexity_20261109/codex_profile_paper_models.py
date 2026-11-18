"""Inference-only complexity measurements. Never train or evaluate the dataset."""
import csv,hashlib,importlib,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net-Repro')
NEW=ROOT.parent/'CLE-Net/latest_top_networks'
RECENT={'mambaout_tiny','mambavision_t','transnext_tiny','thinkingvit_deit','cpubone_b2'}
OUT=Path('/tmp/codex_paper_complexity_20261109')

def repository(r):
    if r['model'] in RECENT:return str(NEW.parent)
    if '/Network/CLE-Net/' in r['source']:return '/home/login/data/cle/auto/Network/CLE-Net'
    if '/ZCode/Classification/' in r['source']:return '/home/login/data/cle/auto/ZCode/Classification'
    if 'cle_net_rerun' in r['source']:return str(NEW.parent)
    return str(ROOT)

def worker(r):
    import torch
    from thop import profile
    torch.set_num_threads(2)
    torch.manual_seed(42)
    repo=repository(r);sys.path.insert(0,repo);os.chdir(repo)
    record=dict(model=r['model'],classes=r['classes'],repo=repo,input_shape=[1,3,224,224],
        dtype='float32',torch=torch.__version__,params=None,macs=None,latency_ms=None,fps=None)
    try:
        if r['model'] in RECENT:
            a=importlib.import_module('latest_top_networks.adapters')
            model=getattr(a,r['model'])(num_classes=r['classes'])
        else:
            a=importlib.import_module('cle_fair_common')
            model=a.build_model(r['model'],r['classes'])
        record['params']=sum(p.numel() for p in model.parameters())
        if r.get('verify_only'):
            d=json.loads(Path(r['source']).read_text())
            checkpoint=Path(d['weights'])
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if isinstance(state,dict) and 'model' in state:state=state['model']
            model.load_state_dict(state,strict=True)
            record['checkpoint_strict_load']=True
            record['checkpoint_params']=sum(state[k].numel() for k,_ in model.named_parameters())
            assert record['params']==record['checkpoint_params']
            return record
        record['parameter_fp32_mib']=record['params']*4/(1024**2)
        record['module_file']=str(Path(sys.modules[model.__class__.__module__].__file__).resolve())
        record['module_sha256']=hashlib.sha256(Path(record['module_file']).read_bytes()).hexdigest()
        model=model.eval().cuda();x=torch.randn(1,3,224,224,device='cuda')
        with torch.inference_mode():
            for _ in range(10):model(x)
            torch.cuda.synchronize()
            samples=[]
            for repeat in range(5):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):model(x)
                end.record();torch.cuda.synchronize();samples.append(start.elapsed_time(end)/20)
        record.update(latency_ms=statistics.mean(samples),latency_sd_ms=statistics.stdev(samples),
            fps=1000/statistics.mean(samples),timing_rounds=5,iterations_per_round=20,warmup=10,
            timing_definition='batch1 CUDA event float32; randomly initialized eval model, no dataset IO or H2D; fps=1000/mean_ms')
        try:
            with torch.inference_mode():macs,params_thop=profile(model,inputs=(x,),verbose=False)
            record.update(macs=float(macs),params_thop=float(params_thop))
        except Exception:record['macs_error']=traceback.format_exc()
        if r['model']=='thinkingvit_deit':
            record['rejected_thop_macs']=record['macs'];record['macs']=None
            record['macs_scope']='THOP invalid for this custom ThinkingViT forward; not reported'
        elif r['model'] in {'mambavision_t','transnext_tiny'} or any(s in r['model'] for s in ['vit_','deit_','swin','pvt_']):
            record['macs_scope']='THOP partial: attention/matrix operations and selective scan may be omitted'
        elif any(s in r['model'] for s in ['egb','esa']):
            record['macs_scope']='THOP partial: FFT/attention and elementwise gates omitted'
        else:record['macs_scope']='THOP supported operators; not total hardware instruction count'
    except Exception:record['error']=traceback.format_exc()
    return record

def main():
    OUT.mkdir(exist_ok=True)
    requests=json.loads(Path('/tmp/codex_complexity_requests.json').read_text())
    grouped={}
    for r in requests:
        key=(repository(r),r['model'],r['classes'])
        grouped.setdefault(key,r)
    profiles=[]
    for i,(key,r) in enumerate(grouped.items()):
        p=OUT/f'profile_{i:03}.json';log=OUT/f'profile_{i:03}.log'
        if p.exists():record=json.loads(p.read_text())
        else:
            # Inference job yields to any already running GPU computation.
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
            with log.open('w') as f:
                result=subprocess.run([sys.executable,__file__,'--worker',json.dumps(r),str(p)],stdout=f,stderr=subprocess.STDOUT,timeout=150)
            if result.returncode:raise RuntimeError(f'Profiler process failed: {log}')
            record=json.loads(p.read_text())
        profiles.append(record)
        print(json.dumps(dict(completed=i+1,total=len(grouped),model=r['model'],classes=r['classes'],params=record['params'],macs=record['macs'],error=record.get('error'))),flush=True)
    combined=[]
    for r in requests:
        key=(repository(r),r['model'],r['classes'])
        item=profiles[list(grouped).index(key)]
        combined.append(dict(**r,measurement=item))
    (OUT/'results.json').write_text(json.dumps(combined,indent=2))
    print('ALL_COMPLETE',len(combined),flush=True)

if __name__=='__main__':
    if '--worker' in sys.argv:
        r=worker(json.loads(sys.argv[2]));Path(sys.argv[3]).write_text(json.dumps(r,indent=2))
    else:main()

import csv,hashlib,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
P=json.loads((ROOT/'protocol.json').read_text())
RUNS=ROOT/'runs'
METRICS=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro','macro_sensitivity','macro_specificity','kappa','mcc']

def status(**kw):
    row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    (ROOT/'status.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)

def preflight():
    import torch,numpy,PIL,torchvision
    from cle_fair_common import build_model
    from train_cle_fair import load_pretrained_weights_strict,set_seed
    torch.set_num_threads(2)
    assert torch.__version__=='2.12.0+cu130'
    manifest=json.loads((ROOT.parent/'CLE-Net-Repro/environment/dataset_manifest.json').read_text())
    for row in manifest:
        f=Path(P['dataset'])/row['split']/row['path']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],f
    audit=dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__,pillow=PIL.__version__,images_verified=len(manifest),seeds=[])
    for seed in P['seeds']:
        set_seed(seed);baseline=build_model(P['models'][0],3)
        load_pretrained_weights_strict(baseline,P['pretrained'],P['models'][0])
        rng=torch.random.get_rng_state().clone()
        set_seed(seed);cle=build_model(P['models'][1],3)
        load_pretrained_weights_strict(cle,P['pretrained'],P['models'][1])
        assert torch.equal(rng,torch.random.get_rng_state())
        for k,v in baseline.state_dict().items():assert torch.equal(v,cle.state_dict()[k]),k
        baseline.eval();cle.eval();x=torch.randn(2,3,224,224)
        with torch.no_grad():
            diff=(baseline(x)-cle(x)).abs().max().item()
        assert diff==0,diff
        logits=cle(x); torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
        gates=[n for n,p in cle.named_parameters() if n.endswith('egb_g') and p.grad is not None and p.grad.abs().sum()>0]
        projections=[n for n,p in cle.named_parameters() if '.scg.c3.weight' in n and p.grad is not None and p.grad.abs().sum()>0]
        assert len(gates)==12 and len(projections)==9,(len(gates),len(projections))
        assert all(torch.isfinite(p.grad).all() for p in cle.parameters() if p.grad is not None)
        audit['seeds'].append(dict(seed=seed,common_state_identical=True,common_rng_identical=True,initial_max_output_difference=diff,
            nonzero_egb_gate_gradients=len(gates),nonzero_scg_projection_gradients=len(projections),
            baseline_params=sum(p.numel() for p in baseline.parameters()),cle_params=sum(p.numel() for p in cle.parameters())))
    (ROOT/'preflight.json').write_text(json.dumps(audit,indent=2))
    status(phase='preflight_passed',audit=audit)

def wait_gpu():
    while True:
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not pids:return
        status(phase='waiting_for_gpu',foreign_pids=pids)
        time.sleep(30)

def execute(command,log,**info):
    wait_gpu()
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        status(pid=child.pid,command=command,**info)
        rc=child.wait()
    if rc:raise RuntimeError(f'{log}: return code {rc}')

def summarize():
    rows=[]
    for seed in P['seeds']:
        for model in P['models']:
            f=RUNS/f'{model}_seed{seed}'/model/'eval_test/results.json'
            if f.exists():
                d=json.loads(f.read_text());assert d['n_total']==576
                rows.append(dict(model=model,seed=seed,source=str(f),**{m:d[m] for m in METRICS}))
    summary=[]
    for model in P['models']:
        rr=[r for r in rows if r['model']==model]
        if not rr:continue
        item=dict(model=model,n=len(rr))
        for m in METRICS:
            vals=[r[m] for r in rr]
            item[m+'_mean']=statistics.mean(vals)
            item[m+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else None
        summary.append(item)
    for name,rr in [('per_seed.csv',rows),('summary.csv',summary)]:
        if rr:
            with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def main():
    preflight()
    if '--check-only' in sys.argv:return
    RUNS.mkdir(exist_ok=True)
    for seed in P['seeds']:
        for model in P['models']:
            base=RUNS/f'{model}_seed{seed}';base.mkdir(exist_ok=True)
            command=[sys.executable,'train_cle_fair.py','--model',model,'--num_classes','3','--img_size','224',
                '--seed',str(seed),'--num-workers','3','--batch-size','32','--lr','0.0001','--epochs','300',
                '--early-stop','30','--label-smoothing','0','--device','cuda:0',
                '--data-path-train',str(Path(P['dataset'])/'train'),'--data-path-val',str(Path(P['dataset'])/'val'),
                '--weights',P['pretrained'],'--save-dir',str(base)]
            run=base/model
            if not (run/'model_best.pth').exists():
                execute(command,base/'train.log',phase='train',seed=seed,model=model)
            else:
                # Resume only a normally completed training, never a partial checkpoint.
                text=(base/'train.log').read_text()
                assert '[early stop]' in text or ('Epoch 299:' in text and 'Training Time:' in text), 'Incomplete training requires separate recovery'
            for split in ('val','test'):
                command=[sys.executable,'evaluate.py','--model',model,'--weights',str(run/'model_best.pth'),
                    '--data-path',str(Path(P['dataset'])/split),'--run-meta',str(run/'run_meta.json'),
                    '--img-size','224','--batch-size','64','--device','cuda:0','--out-dir',str(run/f'eval_{split}')]
                if not (run/f'eval_{split}'/'results.json').exists():
                    execute(command,base/f'eval_{split}.log',phase=split,seed=seed,model=model)
            summary=summarize()
            status(phase='run_complete',model=model,seed=seed,partial_summary=summary)
    status(phase='all_complete',summary=summarize())

if __name__=='__main__':
    try:main()
    except Exception:
        status(phase='failed',traceback=traceback.format_exc())
        raise
